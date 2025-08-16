/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  GenerateContentResponseUsageMetadata,
  ContentListUnion,
  FinishReason,
  Candidate,
  FunctionDeclaration,
  Tool,
  ToolListUnion,
  FunctionCall,
  FunctionResponse,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { setOllamaModelContextLength, getOllamaModelContextLength } from './ollamaTokenLimits.js';

interface OllamaConfig {
  baseUrl: string;
  model: string;
  enableChatApi?: boolean;
  timeout?: number; // Timeout in milliseconds
  streamingTimeout?: number; // Streaming timeout in milliseconds
  contextLimit?: number; // Context window size for conversation tracking
  requestContextSize?: number; // Context window size per request (num_ctx)
  temperature?: number; // Response creativity/randomness (0.0-1.0+)
  debugLogging?: boolean; // Enable debug logging to file
}

interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  format?: string | Record<string, unknown>;
  keep_alive?: string; // Model keep-alive duration
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_predict?: number;
    num_ctx?: number;
    repeat_penalty?: number;
  };
}

interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
}

interface OllamaToolCall {
  id?: string;
  type?: 'function';
  function: {
    name: string;
    arguments: string | Record<string, unknown>; // Can be string or object
  };
}

interface OllamaTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  tools?: OllamaTool[];
  stream?: boolean;
  format?: string | Record<string, unknown>;
  keep_alive?: string; // Model keep-alive duration
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_predict?: number;
    num_ctx?: number;
    repeat_penalty?: number;
  };
}

interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: OllamaChatMessage;
  done: boolean;
  tool_calls?: OllamaToolCall[]; // Tool calls can be at response level
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  tool_calls?: OllamaToolCall[]; // Some models return tool_calls even with /api/generate
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    parent_model: string;
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

interface OllamaListResponse {
  models: OllamaModel[];
}

interface OllamaShowResponse {
  modelfile: string;
  parameters: string;
  template: string;
  details: {
    parent_model: string;
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
  model_info: Record<string, any>;
}

/**
 * ContentGenerator implementation for Ollama API
 */
export class OllamaContentGenerator implements ContentGenerator {
  private config: OllamaConfig;

  constructor(config: OllamaConfig) {
    this.config = config;
    
    // Synchronously start context length initialization but don't wait for it
    this.initializeContextLength();
  }

  /**
   * Write detailed debug log to file for analysis
   */
  private async writeDebugLog(eventType: string, data: any): Promise<void> {
    try {
      const logEntry = {
        timestamp: new Date().toISOString(),
        eventType,
        data,
        separator: '---ENTRY_END---'
      };
      
      const logText = JSON.stringify(logEntry, null, 2) + '\n\n';
      
      const fs = await import('fs');
      const path = await import('path');
      
      // Write to current working directory
      const cwd = process.cwd();
      const ollamaDebugPath = path.join(cwd, 'ollama-debug.log');
      
      console.log(`📝 Writing debug log to: ${ollamaDebugPath}`);
      
      // Write only to ollama-debug.log for Ollama-specific debugging
      fs.appendFileSync(ollamaDebugPath, logText);
      
      // Add performance warnings for slow requests
      if (eventType === 'OLLAMA_RESPONSE' && data.fullResponse?.total_duration) {
        const totalMs = data.fullResponse.total_duration / 1000000; // Convert nanoseconds to milliseconds
        const loadMs = (data.fullResponse.load_duration || 0) / 1000000;
        
        if (totalMs > 10000) { // > 10 seconds
          console.warn(`⚠️  Ollama request very slow (${(totalMs/1000).toFixed(1)}s total, ${(loadMs/1000).toFixed(1)}s loading). Consider using a smaller/faster model.`);
        } else if (loadMs > 3000) { // > 3 seconds loading
          console.warn(`⚠️  Ollama model loading slow (${(loadMs/1000).toFixed(1)}s). Model not kept in memory - consider increasing context limit or using keep_alive.`);
        }
      }
    } catch (error) {
      console.warn('Failed to write debug log:', error);
      console.warn('Error details:', error);
    }
  }

  /**
   * Initialize context length for the model
   */
  private async initializeContextLength(): Promise<void> {
    try {
      const contextLength = await this.getContextLength(this.config.model);
      setOllamaModelContextLength(this.config.model, contextLength);
    } catch (error) {
      console.warn(`Failed to initialize context length for ${this.config.model}:`, error);
      // Set a default value
      setOllamaModelContextLength(this.config.model, 4096);
    }
  }


  /**
   * Ensure context length is initialized before operations
   */
  public async ensureContextLengthInitialized(): Promise<void> {
    // Check if context length is already cached
    if (getOllamaModelContextLength(this.config.model)) {
      return;
    }
    
    // If not cached, initialize it now
    await this.initializeContextLength();
  }

  /**
   * Get request context size (num_ctx parameter for Ollama)
   */
  private getRequestContextSize(): number {
    // Use the dedicated request context size setting
    const requestContextSize = this.config.requestContextSize || 8192;
    
    // Only ensure minimum 1K for safety, no maximum limit
    return Math.max(requestContextSize, 1024);
  }

  /**
   * Estimate token count for text (rough approximation: 1 token ≈ 4 characters)
   */
  private estimateTokenCount(text: string): number {
    if (!text) return 0;
    return Math.ceil(text.length / 4);
  }

  /**
   * Convert Gemini's Tool format to Ollama's tool format
   */
  private convertGeminiToolsToOllama(tools: ToolListUnion): OllamaTool[] {
    const ollamaTools: OllamaTool[] = [];
    
    // Handle different tool union types
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        if (typeof tool === 'object' && 'functionDeclarations' in tool && tool.functionDeclarations) {
          for (const funcDecl of tool.functionDeclarations) {
            // Follow exact Ollama API format from docs
            ollamaTools.push({
              type: 'function',
              function: {
                name: funcDecl.name ?? 'unknown_function',
                description: funcDecl.description ?? '',
                parameters: funcDecl.parametersJsonSchema as Record<string, unknown> || {
                  type: "object",
                  properties: {},
                  required: []
                },
              }
            });
          }
        }
      }
    }
    
    if (this.config.debugLogging) {
      console.log('🔧 Converted Gemini tools to Ollama format:', JSON.stringify(ollamaTools, null, 2));
    }
    
    return ollamaTools;
  }

  /**
   * Convert Gemini-style contents to Ollama messages format
   */
  private contentsToMessages(contents: ContentListUnion): OllamaChatMessage[] {
    if (typeof contents === 'string') {
      return [{ role: 'user', content: contents }];
    }
    
    if (Array.isArray(contents)) {
      if (contents.length === 0) return [];
      
      // Check if it's an array of Content objects (chat history)
      if (typeof contents[0] === 'object' && 'parts' in contents[0]) {
        return this.buildChatMessages(contents as Content[]);
      } else {
        // Array of parts - treat as single user message
        const text = contents
          .map((part) => {
            if (typeof part === 'string') {
              return part;
            }
            if (typeof part === 'object' && 'text' in part) {
              return part.text;
            }
            return '';
          })
          .join('');
        return [{ role: 'user', content: text }];
      }
    }
    
    // Single Content object
    if (typeof contents === 'object' && 'parts' in contents) {
      const content = contents as Content;
      const text = this.extractTextFromContent(content);
      const role = content.role === 'model' ? 'assistant' : 'user';
      return [{ role, content: text }];
    }
    
    return [];
  }

  /**
   * Build chat messages from the complete conversation history
   */
  private buildChatMessages(contents: Content[]): OllamaChatMessage[] {
    const messages: OllamaChatMessage[] = [];
    
    for (const content of contents) {
      if (!content.parts) continue;
      
      // Handle function calls and responses
      const functionCalls: OllamaToolCall[] = [];
      let textContent = '';
      let hasFunctionResponse = false;
      
      for (const part of content.parts) {
        if (typeof part === 'string') {
          textContent += part;
        } else if (typeof part === 'object') {
          if ('text' in part && part.text) {
            textContent += part.text;
          } else if ('functionCall' in part && part.functionCall) {
            const functionCall = part.functionCall as FunctionCall;
            functionCalls.push({
              id: functionCall.id || `call_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              type: 'function',
              function: {
                name: functionCall.name || '',
                arguments: functionCall.args || {},
              }
            });
          } else if ('functionResponse' in part && part.functionResponse) {
            hasFunctionResponse = true;
            // Extract text content from function response for Ollama
            const functionResponse = part.functionResponse as any;
            if (functionResponse.response) {
              // Add the function response content as text
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
        }
      }
      
      if (hasFunctionResponse) {
        // Function responses represent tool execution results
        // Send as 'tool' message as per Ollama API specification
        if (textContent.trim()) {
          messages.push({
            role: 'tool',
            content: textContent.trim(),
          });
        }
      } else {
        // Regular message
        const role = content.role === 'model' ? 'assistant' : 'user';
        const message: OllamaChatMessage = {
          role,
          content: textContent.trim(),
        };
        
        if (functionCalls.length > 0) {
          message.tool_calls = functionCalls;
        }
        
        if (message.content || functionCalls.length > 0) {
          messages.push(message);
        }
      }
    }
    
    return messages;
  }

  /**
   * Extract text content from a Content object
   */
  private extractTextFromContent(content: Content): string {
    if (!content.parts) return '';
    
    return content.parts
      .map((part) => {
        if (typeof part === 'string') {
          return part;
        }
        if (typeof part === 'object' && 'text' in part) {
          return part.text;
        }
        return '';
      })
      .join('')
      .trim();
  }

  /**
   * Convert Gemini-style contents to Ollama prompt (with proper chat history handling)
   */
  private contentsToPrompt(contents: ContentListUnion): string {
    // Handle different content union types
    if (typeof contents === 'string') {
      return contents;
    }
    
    if (Array.isArray(contents)) {
      if (contents.length === 0) return '';
      
      // Check if it's an array of Content objects (chat history)
      if (typeof contents[0] === 'object' && 'parts' in contents[0]) {
        return this.buildChatPrompt(contents as Content[]);
      } else {
        // Array of parts
        return contents
          .map((part) => {
            if (typeof part === 'string') {
              return part;
            }
            if (typeof part === 'object' && 'text' in part) {
              return part.text;
            }
            return '';
          })
          .join('');
      }
    }
    
    // Single Content object
    if (typeof contents === 'object' && 'parts' in contents) {
      const content = contents as Content;
      if (content.parts) {
        return content.parts
          .map((part) => {
            if (typeof part === 'string') {
              return part;
            }
            if (typeof part === 'object' && 'text' in part) {
              return part.text;
            }
            return '';
          })
          .join('');
      }
    }
    
    return '';
  }

  /**
   * Build a chat prompt from the complete conversation history
   */
  private buildChatPrompt(contents: Content[]): string {
    const promptParts: string[] = [];
    
    for (const content of contents) {
      if (!content.parts) continue;
      
      let text = content.parts
        .map((part) => {
          if (typeof part === 'string') {
            return part;
          }
          if (typeof part === 'object' && 'text' in part) {
            return part.text;
          }
          // For now, skip non-text parts (images, function calls, etc.)
          return '';
        })
        .join('')
        .trim();
      
      if (!text) continue;
      
      // Truncate very long system messages to prevent prompt bloat
      if (content.role === 'user' && text.length > 1000) {
        // This is likely the system context - keep only the essential parts
        const lines = text.split('\n');
        const importantLines = lines.filter(line => 
          line.includes('current working directory') ||
          line.includes('Today\'s date') ||
          line.includes('operating system') ||
          line.trim().length < 100 // Keep short lines
        ).slice(0, 10); // Limit to 10 lines max
        
        if (importantLines.length < lines.length) {
          text = importantLines.join('\n') + '\n\n[Context truncated to prevent prompt length issues]';
        }
      }
      
      // Format based on role
      if (content.role === 'user') {
        promptParts.push(`Human: ${text}`);
      } else if (content.role === 'model') {
        promptParts.push(`Assistant: ${text}`);
      } else if (content.role === 'system') {
        promptParts.push(`System: ${text}`);
      } else {
        // Default to user if role is unclear
        promptParts.push(`Human: ${text}`);
      }
    }
    
    // Add a final "Assistant:" to prompt for the next response
    if (promptParts.length > 0) {
      promptParts.push('Assistant:');
    }
    
    return promptParts.join('\n\n');
  }

  /**
   * Build chat messages for API requests - simplified and robust approach
   */
  private buildChatMessagesForApi(request: GenerateContentParameters): OllamaChatMessage[] {
    const messages: OllamaChatMessage[] = [];
    
    // ALWAYS add system message if tools are available - exactly like working cURL
    const toolsSource = (request as any).tools || request.config?.tools;
    const tools = toolsSource ? this.convertGeminiToolsToOllama(toolsSource) : [];
    if (tools.length > 0) {
      messages.push({
        role: 'system',
        content: 'You are a helpful assistant with access to tools.'
      });
    }
    
    // PROPER TOOL CALLING APPROACH: Use full conversation history for tool calling
    // Tool calling requires proper conversation context with assistant/tool message flow
    if (Array.isArray(request.contents)) {
      const conversationMessages = this.buildChatMessages(request.contents as Content[]);
      messages.push(...conversationMessages);
    } else if (typeof request.contents === 'string') {
      messages.push({
        role: 'user',
        content: request.contents
      });
    } else if (request.contents) {
      // Single content object
      const userContent = this.extractTextFromContent(request.contents as any);
      if (userContent) {
        messages.push({
          role: 'user',
          content: userContent
        });
      }
    }
    
    // Debug logging is handled by the main writeDebugLog method
    
    // Validate that we have at least one message
    if (messages.length === 1) {
      // Only system message - need user input
      console.warn('⚠️ No user content found in request, this should not happen!');
      console.log('Request contents:', JSON.stringify(request.contents, null, 2));
      messages.push({
        role: 'user',
        content: "Please provide a valid command to execute."
      });
    }
    
    if (this.config.debugLogging) {
      console.log('🔧 Final chat messages for API:', JSON.stringify(messages, null, 2));
    }
    
    return messages;
  }

  /**
   * Convert Ollama chat response to Gemini-style response
   */
  private ollamaChatToGeminiResponse(
    ollamaResponse: OllamaChatResponse,
    fullPromptText?: string,
  ): GenerateContentResponse {
    const usageMetadata = new GenerateContentResponseUsageMetadata();
    
    // Use Ollama's token counts if available, otherwise estimate
    const promptTokenCount = ollamaResponse.prompt_eval_count || 0;
    const candidatesTokenCount = ollamaResponse.eval_count || 
      this.estimateTokenCount(ollamaResponse.message.content || '');
    
    usageMetadata.promptTokenCount = promptTokenCount;
    usageMetadata.candidatesTokenCount = candidatesTokenCount;
    usageMetadata.totalTokenCount = promptTokenCount + candidatesTokenCount;

    // Convert Ollama message to Gemini parts
    const parts: any[] = [];
    
    // Add text content if present
    if (ollamaResponse.message.content) {
      parts.push({ text: ollamaResponse.message.content });
    }
    
    // Add function calls if present (check both message and response level)
    const toolCalls = ollamaResponse.message.tool_calls || ollamaResponse.tool_calls;
    if (toolCalls) {
      // CRITICAL FIX: Deduplicate tool calls to prevent execution loops
      // Ollama sometimes generates duplicate identical tool calls
      const seenCalls = new Set<string>();
      
      for (const toolCall of toolCalls) {
        let args = {};
        try {
          if (typeof toolCall.function.arguments === 'string') {
            args = JSON.parse(toolCall.function.arguments || '{}');
          } else {
            args = toolCall.function.arguments || {};
          }
        } catch (error) {
          console.warn('Failed to parse tool arguments:', toolCall.function.arguments);
          args = {};
        }
        
        // Create a unique signature for this tool call
        const callSignature = `${toolCall.function.name}:${JSON.stringify(args)}`;
        
        // Skip if we've already seen this exact tool call
        if (seenCalls.has(callSignature)) {
          console.log(`🔄 Skipping duplicate tool call: ${callSignature}`);
          continue;
        }
        seenCalls.add(callSignature);
        
        parts.push({
          functionCall: {
            id: toolCall.id || `call_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: toolCall.function.name,
            args: args,
          }
        });
      }
    }

    const candidate: Candidate = {
      content: {
        parts,
        role: 'model',
      },
      finishReason: ollamaResponse.done ? FinishReason.STOP : undefined,
    };

    const response = new GenerateContentResponse();
    response.candidates = [candidate];
    response.usageMetadata = usageMetadata;

    return response;
  }

  /**
   * Convert Ollama response to Gemini-style response (unified for both APIs)
   */
  private ollamaToGeminiResponse(
    ollamaResponse: OllamaGenerateResponse,
    fullPromptText?: string,
  ): GenerateContentResponse {
    const usageMetadata = new GenerateContentResponseUsageMetadata();
    
    // Use Ollama's token counts if available, otherwise estimate
    // Now promptTokenCount represents the ENTIRE conversation history
    const promptTokenCount = ollamaResponse.prompt_eval_count || 
      (fullPromptText ? this.estimateTokenCount(fullPromptText) : 0);
    const candidatesTokenCount = ollamaResponse.eval_count || 
      this.estimateTokenCount(ollamaResponse.response);
    
    usageMetadata.promptTokenCount = promptTokenCount;
    usageMetadata.candidatesTokenCount = candidatesTokenCount;
    usageMetadata.totalTokenCount = promptTokenCount + candidatesTokenCount;

    // Convert response to parts (text + function calls)
    const parts: any[] = [];
    
    // Add text content if present
    if (ollamaResponse.response) {
      parts.push({ text: ollamaResponse.response });
    }
    
    // Add function calls if present (unified handling for both APIs)
    if (ollamaResponse.tool_calls) {
      // CRITICAL FIX: Deduplicate tool calls to prevent execution loops
      // Ollama sometimes generates duplicate identical tool calls
      const seenCalls = new Set<string>();
      
      for (const toolCall of ollamaResponse.tool_calls) {
        let args = {};
        try {
          if (typeof toolCall.function.arguments === 'string') {
            args = JSON.parse(toolCall.function.arguments || '{}');
          } else {
            args = toolCall.function.arguments || {};
          }
        } catch (error) {
          console.warn('Failed to parse tool arguments:', toolCall.function.arguments);
          args = {};
        }
        
        // Create a unique signature for this tool call
        const callSignature = `${toolCall.function.name}:${JSON.stringify(args)}`;
        
        // Skip if we've already seen this exact tool call
        if (seenCalls.has(callSignature)) {
          console.log(`🔄 Skipping duplicate tool call: ${callSignature}`);
          continue;
        }
        seenCalls.add(callSignature);
        
        parts.push({
          functionCall: {
            id: toolCall.id || `call_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: toolCall.function.name,
            args: args,
          }
        });
      }
    }

    const candidate: Candidate = {
      content: {
        parts,
        role: 'model',
      },
      finishReason: ollamaResponse.done ? FinishReason.STOP : undefined,
    };

    const response = new GenerateContentResponse();
    response.candidates = [candidate];
    response.usageMetadata = usageMetadata;

    return response;
  }

  /**
   * Check if request should use JSON format
   */
  private shouldUseJsonFormat(request: GenerateContentParameters): boolean {
    // Check if the request explicitly asks for JSON response
    return !!(
      request.config?.responseMimeType === 'application/json' ||
      request.config?.responseJsonSchema
    );
  }

  /**
   * Convert JSON schema to a descriptive prompt text
   */
  private schemaToPromptText(schema: any): string {
    if (!schema) return '';
    
    try {
      const schemaStr = JSON.stringify(schema, null, 2);
      return `Please respond in valid JSON format according to the following schema:\n\`\`\`json\n${schemaStr}\n\`\`\`\n\nEnsure your response is valid JSON that conforms to this schema.`;
    } catch (error) {
      return 'Please respond in valid JSON format.';
    }
  }


  /**
   * Extract just the last user message from contents (for large model optimization)
   */
  private getLastUserMessage(contents: ContentListUnion): string {
    if (typeof contents === 'string') {
      return contents;
    }
    
    if (Array.isArray(contents)) {
      // Find the last user message in conversation history
      const contentArray = contents as Content[];
      for (let i = contentArray.length - 1; i >= 0; i--) {
        const content = contentArray[i];
        if (content.role === 'user' && content.parts) {
          const text = content.parts
            .map((part) => {
              if (typeof part === 'string') return part;
              if (typeof part === 'object' && 'text' in part) return part.text;
              return '';
            })
            .join('')
            .trim();
          if (text) return text;
        }
      }
    }
    
    // Fallback to simple extraction
    return this.extractSimpleTextContent(contents);
  }

  /**
   * Simple tool detection for API choice
   */
  private shouldUseChatAPI(hasTools: boolean, enableChatApi: boolean): boolean {
    // Use Chat API only if explicitly enabled AND we have tools
    return enableChatApi && hasTools;
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    if (this.config.debugLogging) {
      console.log(`🚨 OLLAMA generateContent called! PromptID: ${userPromptId}`);
    }
    
    const hasTools = this.hasTools(request);
    const enableChatApi = this.config.enableChatApi ?? true; // Default to Chat API for tool calling
    
    const tools = (request as any).tools || request.config?.tools;
    
    // Detailed debug logging if enabled
    if (this.config.debugLogging) {
      const debugInfo = {
        timestamp: new Date().toISOString(),
        userPromptId,
        hasTools,
        enableChatApi,
        toolsLength: tools ? Array.isArray(tools) ? tools.length : 'not array' : 'no tools',
        contentType: typeof request.contents,
        willUseChatAPI: hasTools && this.shouldUseChatAPI(hasTools, enableChatApi),
        requestContents: Array.isArray(request.contents) ? 
          (request.contents as any[]).map((content, idx) => ({
            index: idx,
            role: content.role,
            partsCount: content.parts?.length || 0,
            hasText: content.parts?.some((p: any) => typeof p === 'string' || p.text),
            hasFunctionCall: content.parts?.some((p: any) => p.functionCall),
            hasFunctionResponse: content.parts?.some((p: any) => p.functionResponse)
          })) : 
          `${typeof request.contents}: ${JSON.stringify(request.contents).substring(0, 100)}...`
      };
      
      try {
        await this.writeDebugLog('GENERATE_CONTENT_CALL', debugInfo);
        console.log('✅ Debug log written successfully');
      } catch (error) {
        console.error('❌ Failed to write debug log:', error);
      }
    }
    
    if (hasTools && this.shouldUseChatAPI(hasTools, enableChatApi)) {
      // Use Chat API for tool calling (as demonstrated in your working cURL example)
      console.log('✅ Ollama generateContent - using Chat API for tool request');
      try {
        return await this.callChatAPI(request, userPromptId);
      } catch (error) {
        console.error('Chat API failed for tools:', error instanceof Error ? error.message : String(error));
        console.debug('Fallback to Generate API - tools will not be executed');
        return await this.callGenerateAPI(request, userPromptId);
      }
    } else {
      // Use Generate API for non-tool requests or when Chat API is disabled
      console.log('🔄 Ollama generateContent - using Generate API');
      return await this.callGenerateAPI(request, userPromptId);
    }
  }

  /**
   * Get timeout for API requests from config
   */
  private getTimeout(): number {
    return this.config.timeout || 120000; // Default 2 minutes if not configured
  }

  /**
   * Get streaming timeout for API requests from config
   */
  private getStreamingTimeout(): number {
    return this.config.streamingTimeout || 300000; // Default 5 minutes if not configured
  }


  /**
   * Call Ollama Chat API (for tool calling) - matches your working cURL example
   */
  private async callChatAPI(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const timeout = this.getTimeout();
    
    // Use AbortController for timeout and cleanup
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn(`Ollama chat request timeout (${timeout}ms), aborting...`);
      controller.abort();
    }, timeout);

    try {
      // Build messages properly handling conversation history
      const messages: OllamaChatMessage[] = this.buildChatMessagesForApi(request);
      
      // Get tools if available
      const toolsSource = (request as any).tools || request.config?.tools;
      const tools = toolsSource ? this.convertGeminiToolsToOllama(toolsSource) : [];
      
      const ollamaRequest: OllamaChatRequest = {
        model: this.config.model,
        messages: messages,
        stream: false, // Non-streaming for single response
      };

      // Add tools if available
      if (tools.length > 0) {
        ollamaRequest.tools = tools;
        console.debug(`Adding ${tools.length} tools to request`);
      }

      // Use request-specific context sizing
      const requestContextSize = this.getRequestContextSize();
      ollamaRequest.options = {
        temperature: this.config.temperature || 0.7,
        num_ctx: requestContextSize,
      };
      
      // Keep model in memory for better performance
      ollamaRequest.keep_alive = '5m'; // Keep model loaded for 5 minutes

      if (this.config.debugLogging) {
        console.log(`🔧 Using request context size: ${requestContextSize} (configured: ${this.config.requestContextSize})`);
      }

      console.log('🔍 Ollama chat request being sent:', {
        model: ollamaRequest.model,
        messageCount: ollamaRequest.messages.length,
        toolCount: tools.length,
        timeoutMs: timeout,
        hasStream: ollamaRequest.stream === false,
      });
      
      if (this.config.debugLogging) {
        console.log('🔍 Full request JSON:');
        console.log(JSON.stringify(ollamaRequest, null, 2));
      }
      
      // Debug log the request if enabled
      if (this.config.debugLogging) {
        await this.writeDebugLog('OLLAMA_REQUEST', {
          model: ollamaRequest.model,
          messagesCount: ollamaRequest.messages.length,
          toolsCount: ollamaRequest.tools?.length || 0,
          fullRequest: ollamaRequest
        });
      }
      
      // Additional validation to prevent GPU hangs
      if (ollamaRequest.messages.length === 0) {
        throw new Error('No messages to send to Ollama API');
      }
      
      // Check for excessively long content that might cause GPU hangs
      const totalContentLength = ollamaRequest.messages
        .map(msg => msg.content?.length || 0)
        .reduce((sum, len) => sum + len, 0);
      
      if (totalContentLength > 10000) {
        console.warn(`⚠️ Large content detected (${totalContentLength} chars) - may cause GPU hang`);
      }
      
      console.log('🌐 Making fetch request to Ollama...');
      const response = await fetch(`${this.config.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
        signal: controller.signal,
      });

      console.log('📡 Fetch response received:', response.status, response.statusText);

      if (!response.ok) {
        clearTimeout(timeoutId);
        const errorText = await response.text();
        throw new Error(`Ollama API error ${response.status}: ${errorText}`);
      }

      console.log('🔄 Reading response as text first...');
      const responseText = await response.text();
      console.log('📄 Response text length:', responseText.length);
      console.log('📄 Response preview:', responseText.substring(0, 200));
      
      console.log('🔄 Parsing JSON...');
      const ollamaResponse: OllamaChatResponse = JSON.parse(responseText);
      console.log('✅ JSON parsed successfully');

      clearTimeout(timeoutId);
      console.debug('Ollama chat response received:', {
        hasMessage: !!ollamaResponse.message,
        hasContent: !!ollamaResponse.message?.content,
        hasToolCalls: !!ollamaResponse.message?.tool_calls,
        done: ollamaResponse.done
      });
      
      // Debug log the response if enabled
      if (this.config.debugLogging) {
        await this.writeDebugLog('OLLAMA_RESPONSE', {
          hasMessage: !!ollamaResponse.message,
          hasContent: !!ollamaResponse.message?.content,
          hasToolCalls: !!ollamaResponse.message?.tool_calls,
          toolCallsCount: ollamaResponse.message?.tool_calls?.length || 0,
          toolCalls: ollamaResponse.message?.tool_calls?.map(call => ({
            name: call.function.name,
            arguments: call.function.arguments,
            id: call.id
          })),
          done: ollamaResponse.done,
          fullResponse: ollamaResponse
        });
      }
      
      return this.ollamaChatToGeminiResponse(ollamaResponse);
    } catch (error) {
      clearTimeout(timeoutId);
      
      const err = error as Error;
      
      if (err.name === 'AbortError') {
        throw new Error(`Ollama chat request was aborted due to timeout (${timeout}ms)`);
      }
      
      throw new Error(`Failed to generate content with Ollama tools: ${err.message}`);
    }
  }

  /**
   * Check if request has tools (like Gemini pattern)
   */
  private hasTools(request: GenerateContentParameters): boolean {
    // Check both request.tools and request.config?.tools for compatibility
    // Note: TypeScript may not recognize 'tools' property, but it exists in the runtime API
    const tools = (request as any).tools || request.config?.tools;
    return !!(tools && Array.isArray(tools) && tools.length > 0);
  }

  /**
   * Extract simple text content from ContentListUnion (safe version)
   */
  private extractSimpleTextContent(contents: ContentListUnion): string {
    if (typeof contents === 'string') {
      return contents;
    }
    
    if (Array.isArray(contents)) {
      return contents
        .map((item) => {
          if (typeof item === 'string') return item;
          if (typeof item === 'object' && 'text' in item) return item.text;
          return '';
        })
        .join(' ')
        .trim();
    }
    
    if (typeof contents === 'object' && 'parts' in contents) {
      return this.extractTextFromContent(contents as Content);
    }
    
    return '';
  }

  /**
   * Call Ollama Generate API (standard generation with tool support)
   */
  private async callGenerateAPI(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    let prompt = this.contentsToPrompt(request.contents);
    
    const ollamaRequest: OllamaGenerateRequest = {
      model: this.config.model,
      prompt: prompt,
      stream: false,
    };

    // Note: Generate API does not support tools (per Ollama docs)
    // Tools are only supported by Chat API

    // Handle JSON schema requests
    if (this.shouldUseJsonFormat(request)) {
      if (request.config?.responseJsonSchema) {
        // Use the actual schema as the format parameter according to Ollama API docs
        ollamaRequest.format = request.config.responseJsonSchema as Record<string, unknown>;
        // Also add explicit instructions to the prompt
        const schemaPrompt = this.schemaToPromptText(request.config.responseJsonSchema);
        ollamaRequest.prompt = prompt + '\n\n' + schemaPrompt + '\n\nRespond with valid JSON only, no additional text or formatting.';
      } else {
        ollamaRequest.format = 'json';
        ollamaRequest.prompt = prompt + '\n\nRespond with valid JSON only, no additional text or formatting.';
      }
    }

    // Apply basic generation config with performance optimizations
    const requestContextSize = this.getRequestContextSize();
    ollamaRequest.options = {
      temperature: this.config.temperature || 0.7,
      num_ctx: requestContextSize,
    };
    
    // Keep model in memory for better performance
    ollamaRequest.keep_alive = '5m'; // Keep model loaded for 5 minutes

    if (this.config.debugLogging) {
      console.log(`🔧 Using request context size: ${requestContextSize} (configured: ${this.config.requestContextSize})`);
    }

    try {
      console.debug('Ollama request:', JSON.stringify(ollamaRequest, null, 2));
      
      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const ollamaResponse: OllamaGenerateResponse = await response.json();
      console.debug('Ollama response:', JSON.stringify(ollamaResponse, null, 2));
      
      // Check if we got a valid response (either text or tool calls)
      if ((!ollamaResponse.response || ollamaResponse.response.trim() === '') && 
          (!ollamaResponse.tool_calls || ollamaResponse.tool_calls.length === 0)) {
        throw new Error('Ollama returned an empty response. This may indicate the model cannot handle the requested JSON schema or the prompt is too complex.');
      }
      
      return this.ollamaToGeminiResponse(ollamaResponse, prompt);
    } catch (error) {
      throw new Error(`Failed to generate content with Ollama: ${error}`);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    if (this.config.debugLogging) {
      console.log(`🔄 OLLAMA generateContentStream called! PromptID: ${userPromptId}`);
    }
    
    const hasTools = this.hasTools(request);
    const enableChatApi = this.config.enableChatApi ?? true;
    
    if (hasTools && this.shouldUseChatAPI(hasTools, enableChatApi)) {
      // Use Chat API for tool calling
      if (this.config.debugLogging) {
        console.log(`✅ Ollama generateContentStream - using Chat API for tool request`);
      }
      try {
        return this.callChatAPIStream(request, userPromptId);
      } catch (error) {
        console.warn('Chat API stream failed for tools:', error instanceof Error ? error.message : String(error));
        console.debug('Fallback to Generate API stream - tools will not be executed');
        return this.callGenerateAPIStream(request, userPromptId);
      }
    } else {
      // Use Generate API for non-tool requests or when Chat API is disabled
      if (this.config.debugLogging) {
        console.log(`🔄 Ollama generateContentStream - using Generate API`);
      }
      return this.callGenerateAPIStream(request, userPromptId);
    }
  }

  /**
   * Call Ollama Chat API Stream (for tool calling) - streaming version
   */
  private async *callChatAPIStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const timeout = this.getStreamingTimeout(); // Use dedicated streaming timeout
    
    // Use AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn(`Ollama chat stream timeout (${timeout}ms), aborting...`);
      controller.abort();
    }, timeout);

    try {
      // Debug log the stream call if enabled
      if (this.config.debugLogging) {
        await this.writeDebugLog('CHAT_API_STREAM_START', {
          userPromptId,
          timeout,
          hasTools: this.hasTools(request),
          contentType: typeof request.contents,
          isArray: Array.isArray(request.contents),
          arrayLength: Array.isArray(request.contents) ? request.contents.length : 0
        });
      }

      // Build messages using the improved method
      const messages: OllamaChatMessage[] = this.buildChatMessagesForApi(request);
      const toolsSource = (request as any).tools || request.config?.tools;
      const tools = toolsSource ? this.convertGeminiToolsToOllama(toolsSource) : [];
      
      const ollamaRequest: OllamaChatRequest = {
        model: this.config.model,
        messages: messages,
        stream: true, // Enable streaming
      };

      // Add tools if available
      if (tools.length > 0) {
        ollamaRequest.tools = tools;
      }

      // Use request-specific context sizing
      const requestContextSize = this.getRequestContextSize();
      ollamaRequest.options = {
        temperature: this.config.temperature || 0.7,
        num_ctx: requestContextSize,
      };
      
      // Keep model in memory for better performance
      ollamaRequest.keep_alive = '5m'; // Keep model loaded for 5 minutes

      if (this.config.debugLogging) {
        console.log(`🔧 Using request context size: ${requestContextSize} (configured: ${this.config.requestContextSize})`);
      }

      // Debug log the request if enabled
      if (this.config.debugLogging) {
        await this.writeDebugLog('CHAT_API_STREAM_REQUEST', {
          model: ollamaRequest.model,
          messagesCount: ollamaRequest.messages.length,
          toolsCount: ollamaRequest.tools?.length || 0,
          fullRequest: ollamaRequest
        });
      }

      console.log(`🚀 Ollama Chat API stream request to: ${this.config.baseUrl}/api/chat`);
      const response = await fetch(`${this.config.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API error ${response.status}: ${errorText}`);
      }

      if (!response.body) {
        throw new Error('No response body from Ollama API');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let hasYieldedAnyContent = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim()) {
              try {
                const ollamaResponse: OllamaChatResponse = JSON.parse(line);
                
                // Only yield if there's new content or tool calls
                if ((ollamaResponse.message && ollamaResponse.message.content) || 
                    (ollamaResponse.message && ollamaResponse.message.tool_calls)) {
                  const geminiResponse = this.ollamaChatToGeminiResponse(ollamaResponse);
                  
                  // For streaming, only include the new incremental text
                  if (geminiResponse.candidates && geminiResponse.candidates[0] && 
                      geminiResponse.candidates[0].content && geminiResponse.candidates[0].content.parts &&
                      geminiResponse.candidates[0].content.parts[0] && 
                      'text' in geminiResponse.candidates[0].content.parts[0] &&
                      ollamaResponse.message.content) {
                    // Send incremental content
                    geminiResponse.candidates[0].content.parts[0].text = ollamaResponse.message.content;
                  }
                  
                  yield geminiResponse;
                  hasYieldedAnyContent = true;
                }

                if (ollamaResponse.done) {
                  break;
                }
              } catch (parseError) {
                console.warn('Failed to parse Ollama chat stream response:', parseError);
                console.warn('Problematic line:', line);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
        clearTimeout(timeoutId);
      }
    } catch (error) {
      clearTimeout(timeoutId);
      
      const err = error as Error;
      
      if (err.name === 'AbortError') {
        throw new Error(`Ollama chat stream was aborted due to timeout (${timeout}ms)`);
      }
      
      throw new Error(`Failed to generate content stream with Ollama tools: ${err.message}`);
    }
  }

  private async *callGenerateAPIStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse> {
    let prompt = this.contentsToPrompt(request.contents);
    const originalPrompt = prompt;
    
    const ollamaRequest: OllamaGenerateRequest = {
      model: this.config.model,
      prompt: prompt,
      stream: true,
    };

    // Note: Generate API does not support tools (per Ollama docs)
    // Tools are only supported by Chat API

    // Handle JSON schema requests
    if (this.shouldUseJsonFormat(request)) {
      if (request.config?.responseJsonSchema) {
        // Use the actual schema as the format parameter according to Ollama API docs
        ollamaRequest.format = request.config.responseJsonSchema as Record<string, unknown>;
        // Also add explicit instructions to the prompt
        const schemaPrompt = this.schemaToPromptText(request.config.responseJsonSchema);
        ollamaRequest.prompt = prompt + '\n\n' + schemaPrompt + '\n\nRespond with valid JSON only, no additional text or formatting.';
      } else {
        ollamaRequest.format = 'json';
        ollamaRequest.prompt = prompt + '\n\nRespond with valid JSON only, no additional text or formatting.';
      }
    }

    // Apply basic generation config with performance optimizations
    const requestContextSize = this.getRequestContextSize();
    ollamaRequest.options = {
      temperature: this.config.temperature || 0.7,
      num_ctx: requestContextSize,
    };
    
    // Keep model in memory for better performance
    ollamaRequest.keep_alive = '5m'; // Keep model loaded for 5 minutes

    if (this.config.debugLogging) {
      console.log(`🔧 Using request context size: ${requestContextSize} (configured: ${this.config.requestContextSize})`);
    }

    try {
      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No response body from Ollama API');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let totalResponse = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim()) {
              try {
                const ollamaResponse: OllamaGenerateResponse = JSON.parse(line);
                
                // Only yield if there's new content
                if (ollamaResponse.response) {
                  totalResponse += ollamaResponse.response;
                  
                  const geminiResponse = this.ollamaToGeminiResponse(ollamaResponse, originalPrompt);
                  // For streaming, only include the new incremental text
                  if (geminiResponse.candidates && geminiResponse.candidates[0] && 
                      geminiResponse.candidates[0].content && geminiResponse.candidates[0].content.parts &&
                      geminiResponse.candidates[0].content.parts[0] && 
                      'text' in geminiResponse.candidates[0].content.parts[0]) {
                    // Only send the new incremental content, not the cumulative content
                    geminiResponse.candidates[0].content.parts[0].text = ollamaResponse.response;
                  }
                  
                  yield geminiResponse;
                }

                if (ollamaResponse.done) {
                  break;
                }
              } catch (parseError) {
                console.warn('Failed to parse Ollama stream response:', parseError);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      throw new Error(`Failed to generate content stream with Ollama: ${error}`);
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Ollama doesn't have a direct token counting API
    // We'll estimate based on text length (rough approximation: 1 token ≈ 4 characters)
    const prompt = this.contentsToPrompt(request.contents);
    const estimatedTokens = Math.ceil(prompt.length / 4);
    
    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Ollama has embedding support, but for now we'll throw an error
    // This can be implemented later if needed
    throw new Error('Embedding not implemented for Ollama yet');
  }

  /**
   * Get available models from Ollama
   */
  async listModels(): Promise<OllamaModel[]> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`);
      if (!response.ok) {
        throw new Error(`Failed to list models: ${response.status} ${response.statusText}`);
      }
      const data: OllamaListResponse = await response.json();
      return data.models;
    } catch (error) {
      throw new Error(`Failed to list Ollama models: ${error}`);
    }
  }

  /**
   * Get model information including context length
   */
  async getModelInfo(modelName: string): Promise<OllamaShowResponse> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/show`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: modelName }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.status} ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      throw new Error(`Failed to get Ollama model info: ${error}`);
    }
  }

  /**
   * Extract context length from model info
   */
  async getContextLength(modelName: string): Promise<number> {
    try {
      const modelInfo = await this.getModelInfo(modelName);
      
      // Look for context_length in model_info
      for (const [key, value] of Object.entries(modelInfo.model_info || {})) {
        if (key.endsWith('context_length')) {
          return typeof value === 'number' ? value : parseInt(value as string, 10);
        }
      }
      
      // Default context length if not found
      return 2048;
    } catch (error) {
      console.warn(`Failed to get context length for ${modelName}:`, error);
      return 2048; // Default fallback
    }
  }
}