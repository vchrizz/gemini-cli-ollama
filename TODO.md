# Tool-Calling Loop Problem Analysis & Fix Attempts

## Problem Description
When executing a tool command like `run date command` in Ollama integration:
1. ✅ Tool is executed successfully (e.g., date command runs)
2. ✅ Result is displayed correctly
3. ❌ **LOOP**: Same tool execution request appears again asking for permission
4. ❌ If "allow always" is selected, the command executes in infinite loop

## Root Cause Analysis

### Initial Investigation
- **NOT a duplicate tool-call issue**: Ollama generates correct single tool-calls
- **NOT an OllamaContentGenerator issue**: The streaming and response handling works
- **NOT a settings/config issue**: Ollama integration works for non-tool requests

### Real Root Cause Identified
**Problem Location**: `useGeminiStream.ts` line 841-847 in `handleCompletedTools()`

**The Issue**: After tool execution, `submitQuery()` automatically sends tool results back to Ollama as `functionResponse` parts. However, `buildChatMessages()` in OllamaContentGenerator doesn't properly handle `functionResponse` parts, causing conversation context confusion.

**Technical Details**:
1. Tool executes → creates `functionResponse` part
2. `handleCompletedTools()` calls `submitQuery(mergePartListUnions(responsesToSend))`
3. `responsesToSend` contains `functionResponse` objects like:
   ```typescript
   {
     functionResponse: {
       name: 'run_shell_command',
       id: 'call_123',
       response: { output: 'Thu Aug 15 01:30:15 UTC 2024' }
     }
   }
   ```
4. `buildChatMessages()` line 333-336 ignores `functionResponse` parts
5. Ollama doesn't receive proper conversation context
6. Ollama regenerates the same tool call → LOOP

## Fix Attempts

### Attempt 1: Tool Call Deduplication ❌
**Approach**: Added duplicate detection in tool call processing
**Location**: `ollamaContentGenerator.ts` lines 605-640
**Result**: Fixed symptom but not root cause

### Attempt 2: Conversation History Simplification ❌  
**Approach**: Only send latest user message to avoid confusion
**Location**: `buildChatMessagesForApi()` method
**Result**: Broke conversation context

### Attempt 3: Function Response Processing Fix ✅ (Partial)
**Approach**: Properly extract and handle `functionResponse` parts
**Location**: `buildChatMessages()` lines 333-349
**Changes Made**:
```typescript
// OLD (ignored functionResponse):
} else if ('functionResponse' in part && part.functionResponse) {
  hasFunctionResponse = true;
  // For function responses, we'll handle them separately
}

// NEW (extracts content):
} else if ('functionResponse' in part && part.functionResponse) {
  hasFunctionResponse = true;
  const functionResponse = part.functionResponse as any;
  if (functionResponse.response) {
    if (typeof functionResponse.response === 'string') {
      textContent += functionResponse.response;
    } else if (functionResponse.response.output) {
      textContent += functionResponse.response.output;
    } else if (functionResponse.response.error) {
      textContent += `Error: ${functionResponse.response.error}`;
    } else {
      textContent += JSON.stringify(functionResponse.response);
    }
  }
}
```

**Result**: Improved but problem persists

## Current Status
- ✅ Debug logging implemented and working
- ✅ Function response extraction implemented  
- ✅ Tool deduplication as safety net
- ✅ **FIXED: Tool execution loop problem resolved**

## ✅ SOLUTION IMPLEMENTED

### Final Fix: Tool Role Correction ✅
**Problem**: Function responses were being sent as `user` messages instead of `tool` messages
**Location**: `buildChatMessages()` line 358 in `ollamaContentGenerator.ts`
**Solution**: Changed `role: 'user'` to `role: 'tool'` for function responses

**Root Cause**: The Ollama API expects tool execution results to use the `tool` role, not `user` role:
- ✅ **Correct flow**: `user` → `assistant` (with tool_calls) → `tool` (results) → `assistant` (final response)
- ❌ **Wrong flow**: `user` → `assistant` (with tool_calls) → `user` (results) ← **This caused the loop**

**Technical Details**:
- According to official Ollama documentation, tool responses must use `role: 'tool'`
- When tool responses were sent as `user` messages, Ollama interpreted them as new user requests
- This caused Ollama to regenerate the same tool call, creating an infinite loop

**Verification**: Test case confirms function responses now correctly use `tool` role

### Additional Fix: Tool Arguments Format ✅
**Problem**: Ollama API error 400 - arguments were sent as JSON string instead of object
**Location**: `buildChatMessages()` line 330 in `ollamaContentGenerator.ts`
**Error**: `json: cannot unmarshal string into Go struct field ChatRequest.messages.tool_calls.function.arguments of type api.ToolCallFunctionArguments`
**Solution**: Changed `arguments: JSON.stringify(functionCall.args || {})` to `arguments: functionCall.args || {}`

**Technical Details**:
- Ollama API expects `tool_calls.function.arguments` as a JavaScript object
- Previous implementation converted to JSON string, causing Go unmarshaling error
- TypeScript interface already supported both string and object types

**Final Status**: ✅ **COMPLETELY RESOLVED** - Tool calling works without loops or API errors

## Debug Information Available
- `ollama-debug.log`: Contains detailed request/response logs when `ollamaDebugLogging: true`
- Debug logs show correct tool detection and extraction
- Problem reproduction is consistent and reliable

## Files Modified
- `packages/core/src/core/ollamaContentGenerator.ts`: Function response handling, debug logging
- `test_loop_fix.js`: Test script for reproduction (can be deleted)

## Key Insight
The issue is **NOT** in Ollama response processing but in how **function responses are submitted back to Ollama** in the conversation continuation flow. The automatic `submitQuery()` after tool completion needs investigation.