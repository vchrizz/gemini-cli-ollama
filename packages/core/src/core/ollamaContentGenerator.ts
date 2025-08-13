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
  gpuHangProtection?: boolean;
  modelSizeThreshold?: number; // GB - models above this size get extra protection
}

interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  format?: string | Record<string, unknown>;
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
  private gpuHangDetected: boolean = false;
  private consecutiveTimeouts: number = 0;
  private lastModelSize?: number; // Track model size for hang protection

  constructor(config: OllamaConfig) {
    this.config = config;
    // Enable GPU hang protection by default
    this.config.gpuHangProtection = this.config.gpuHangProtection ?? true;
    this.config.modelSizeThreshold = this.config.modelSizeThreshold ?? 30; // 30GB threshold
    
    // Synchronously start context length initialization but don't wait for it
    this.initializeContextLength();
    // Also initialize model size detection
    this.initializeModelSize();
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
   * Initialize model size detection for GPU hang protection
   */
  private async initializeModelSize(): Promise<void> {
    if (!this.config.gpuHangProtection) return;
    
    try {
      const models = await this.listModels();
      const currentModel = models.find(m => m.name === this.config.model || m.model === this.config.model);
      if (currentModel) {
        // Convert bytes to GB
        this.lastModelSize = currentModel.size / (1024 * 1024 * 1024);
        console.debug(`Model ${this.config.model} size: ${this.lastModelSize.toFixed(1)}GB`);
        
        if (this.lastModelSize > (this.config.modelSizeThreshold || 30)) {
          console.warn(`⚠️  Large model detected (${this.lastModelSize.toFixed(1)}GB). GPU hang protection enabled.`);
        }
      }
    } catch (error) {
      console.warn('Failed to detect model size for GPU hang protection:', error);
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
            ollamaTools.push({
              type: 'function',
              function: {
                name: funcDecl.name ?? 'unknown_function',
                description: funcDecl.description ?? '',
                parameters: funcDecl.parametersJsonSchema as Record<string, unknown> || {},
              }
            });
          }
        }
      }
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
                arguments: JSON.stringify(functionCall.args || {}),
              }
            });
          } else if ('functionResponse' in part && part.functionResponse) {
            hasFunctionResponse = true;
            // For function responses, we'll handle them separately
          }
        }
      }
      
      if (hasFunctionResponse) {
        // This is a tool response message
        messages.push({
          role: 'tool',
          content: textContent.trim() || 'Function executed successfully',
        });
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
   * Convert Ollama chat response to Gemini-style response
   */
  private ollamaChatToGeminiResponse(
    ollamaResponse: OllamaChatResponse,
    fullPromptText?: string,
  ): GenerateContentResponse {
    const usageMetadata = new GenerateContentResponseUsageMetadata();
    
    // Use Ollama's token counts if available, otherwise estimate
    const promptTokenCount = ollamaResponse.prompt_eval_count || 
      (fullPromptText ? this.estimateTokenCount(fullPromptText) : 0);
    const candidatesTokenCount = ollamaResponse.eval_count || 
      this.estimateTokenCount(ollamaResponse.message.content);
    
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
   * Intelligent model assessment for Chat API compatibility
   */
  private shouldUseChatAPI(hasTools: boolean, enableChatApi: boolean): { shouldUse: boolean; reason: string } {
    const modelName = this.config.model.toLowerCase();
    const modelSize = this.lastModelSize || 0;
    
    // If Chat API is disabled, always use Generate API
    if (!enableChatApi) {
      return { shouldUse: false, reason: 'Chat API disabled in settings' };
    }
    
    // If no tools, prefer Generate API (more stable)
    if (!hasTools) {
      return { shouldUse: false, reason: 'No tools requested, Generate API preferred' };
    }
    
    // Known problematic models with streaming issues
    const streamingProblematic = ['qwen2.5', 'qwen', 'yi-', 'deepseek-r1'];
    if (streamingProblematic.some(model => modelName.includes(model))) {
      return { shouldUse: false, reason: `Model ${modelName} has known streaming issues with Chat API` };
    }
    
    // GPU hang protection for large models
    if (modelSize > 8) {
      if (this.consecutiveTimeouts >= 2) {
        return { shouldUse: false, reason: `Large model (${modelSize.toFixed(1)}GB) with persistent issues` };
      }
      console.warn(`⚠️ WARNING: Large model (${modelSize.toFixed(1)}GB) with Chat API - GPU hangs possible!`);
    }
    
    // After consecutive timeouts, switch to Generate API
    if (this.consecutiveTimeouts >= 2) {
      return { shouldUse: false, reason: 'Too many consecutive failures, fallback to Generate API' };
    }
    
    return { shouldUse: true, reason: 'Model appears compatible with Chat API' };
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const hasTools = this.hasTools(request);
    const enableChatApi = this.config.enableChatApi ?? false;
    
    // INTELLIGENT MODEL ASSESSMENT: Check if model is suitable for Chat API
    const chatApiAssessment = this.shouldUseChatAPI(hasTools, enableChatApi);
    
    if (hasTools) {
      // For tool requests: Check if Chat API is safe to use
      console.debug('Ollama generateContent - tool request detected');
      
      if (!chatApiAssessment.shouldUse) {
        console.warn(`🛡️ Skipping Chat API: ${chatApiAssessment.reason}`);
        console.warn('Tools will not execute but system remains stable');
        return await this.callGenerateAPI(request, userPromptId);
      }
      
      console.debug(`✅ Using Chat API: ${chatApiAssessment.reason}`);
      
      try {
        return await this.callChatAPI(request, userPromptId);
      } catch (error) {
        this.detectGpuHang(error instanceof Error ? error : new Error(String(error)));
        console.error('Chat API failed for tools:', error instanceof Error ? error.message : String(error));
        console.debug('Emergency fallback to Generate API - tools will not be executed');
        return await this.callGenerateAPI(request, userPromptId);
      }
    } else {
      // For non-tool requests: Generate API (stable and fast)
      console.debug('Ollama generateContent - using generate API for non-tool request');
      return await this.callGenerateAPI(request, userPromptId);
    }
  }

  /**
   * Get timeout value based on model size and GPU hang protection
   */
  private getChatTimeout(): number {
    if (!this.config.gpuHangProtection) return 15000;
    
    const modelSize = this.lastModelSize || 0;
    const modelName = this.config.model.toLowerCase();
    
    // EXPANDED PROTECTION: Cover both GPU hangs AND streaming problems
    
    // Known problematic models with streaming issues
    const streamingProblematicModels = ['qwen2.5', 'qwen', 'yi-'];
    const hasStreamingIssues = streamingProblematicModels.some(model => modelName.includes(model));
    
    // GPU hang protection for large models (> 8GB)
    if (modelSize > 8) {
      const baseTimeout = 5000; // Very short base timeout
      const timeout = Math.max(3000, baseTimeout - (this.consecutiveTimeouts * 1000));
      console.debug(`🛡️ GPU hang protection: ${timeout}ms timeout for ${modelSize.toFixed(1)}GB model`);
      return timeout;
    }
    
    // Streaming problem protection for known problematic models
    if (hasStreamingIssues || this.consecutiveTimeouts > 0) {
      const baseTimeout = 8000; // Medium timeout for streaming issues
      const timeout = Math.max(5000, baseTimeout - (this.consecutiveTimeouts * 1000));
      console.debug(`🛡️ Streaming protection: ${timeout}ms timeout for model ${modelName}`);
      return timeout;
    }
    
    // Medium-large models (4-8GB) get moderate protection
    if (modelSize > 4) {
      return 12000; // Slightly longer timeout
    }
    
    return 15000; // Default timeout for small models
  }

  /**
   * Detect GPU hang patterns and streaming problems, adjust strategy accordingly
   */
  private detectGpuHang(error: Error): boolean {
    const errorMessage = error.message.toLowerCase();
    const modelSize = this.lastModelSize || 0;
    const modelName = this.config.model.toLowerCase();
    
    // GPU hang indicators (hardware issues)
    const gpuHangIndicators = [
      'gpu hang',
      'page fault', 
      'queue eviction',
      'mes failed',
      'mode2 reset',
      'unrecoverable state',
      'hw exception',
      'vram usage didn\'t recover',
      'signal: aborted',
      'core dumped'
    ];
    
    // Streaming problem indicators (software issues)
    const streamingProblemIndicators = [
      'aborted due to timeout',
      'request was aborted',
      'timeout',
      'context canceled'
    ];
    
    const isGpuHang = gpuHangIndicators.some(indicator => errorMessage.includes(indicator));
    const isStreamingProblem = streamingProblemIndicators.some(indicator => errorMessage.includes(indicator)) || error.name === 'AbortError';
    
    if (isGpuHang || isStreamingProblem) {
      this.consecutiveTimeouts++;
      this.gpuHangDetected = true;
      
      if (isGpuHang) {
        console.error(`🚨 ROCm GPU HANG detected (${this.consecutiveTimeouts} consecutive)!`);
        console.error(`🔥 Model: ${this.config.model} (${modelSize.toFixed(1)}GB) - Error: ${error.message}`);
        
        // GPU hang specific recommendations
        if (this.consecutiveTimeouts === 1) {
          console.error('💡 SOLUTION: Use smaller model for tool calling to avoid ROCm GPU hangs:');
          console.error('   ollama pull llama3.2:3b   # 2GB model, very stable');
          console.error('   ollama pull llama3:8b      # 4.7GB model, stable');
        }
      } else if (isStreamingProblem) {
        console.warn(`⚠️ STREAMING PROBLEM detected (${this.consecutiveTimeouts} consecutive)`);
        console.warn(`🔄 Model: ${this.config.model} (${modelSize.toFixed(1)}GB) - Timeout/Loop issue`);
        
        // Streaming problem specific recommendations
        if (this.consecutiveTimeouts === 1) {
          console.warn('💡 SOLUTION: Model has Chat API streaming issues. Switching to Generate API:');
          console.warn('   - Tool calling will be disabled but system remains stable');
          console.warn('   - Consider using different model if tool calling is needed');
        }
      }
      
      // After 2 consecutive issues, disable chat API entirely
      if (this.consecutiveTimeouts >= 2) {
        const problemType = isGpuHang ? 'GPU hangs' : 'streaming problems';
        console.error(`🚫 CRITICAL: Disabling Chat API due to persistent ${problemType}`);
        console.error('Tools will use Generate API (limited functionality but stable)');
      }
      
      return true;
    }
    
    // Reset on successful operation
    if (this.consecutiveTimeouts > 0) {
      console.info(`✅ Recovery successful after ${this.consecutiveTimeouts} attempts`);
      this.consecutiveTimeouts = 0;
      this.gpuHangDetected = false;
    }
    
    return false;
  }

  /**
   * Call Ollama Chat API (for tool calling) with GPU hang protection
   */
  private async callChatAPI(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const timeout = this.getChatTimeout();
    
    // Use AbortController for timeout and cleanup
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn(`Ollama chat request timeout (${timeout}ms), aborting...`);
      controller.abort();
    }, timeout);

    try {
      // Simplified message conversion to avoid complex parsing issues
      const userMessage = this.extractSimpleTextContent(request.contents);
      const tools = request.config?.tools ? this.convertGeminiToolsToOllama(request.config.tools) : [];
      
      const ollamaRequest: OllamaChatRequest = {
        model: this.config.model,
        messages: [{ role: 'user', content: userMessage }], // Keep it simple
        stream: false,
      };

      // Add tools if available
      if (tools.length > 0) {
        ollamaRequest.tools = tools;
        console.debug(`Adding ${tools.length} tools to request`);
      }

      // CRITICAL: Ultra-aggressive GPU hang protection based on journal evidence
      const modelSize = this.lastModelSize || 0;
      const isProblematicModel = modelSize > 8; // Even 12.8GB gpt-oss:20b causes hangs
      
      // Severely limit tokens for any model > 8GB to prevent VRAM issues
      let maxTokens = 2048;
      if (isProblematicModel) {
        maxTokens = this.gpuHangDetected ? 256 : 512; // Very small responses
      }
      if (this.consecutiveTimeouts > 0) {
        maxTokens = Math.max(128, maxTokens / 2); // Even smaller after timeouts
      }
      
      ollamaRequest.options = {
        temperature: request.config?.temperature || 0.7,
        num_predict: maxTokens,
        // Additional ROCm stability options
        num_ctx: 2048, // Smaller context to reduce VRAM pressure
        repeat_penalty: 1.1, // Prevent generation loops that can trigger hangs
      };
      
      console.debug(`🛡️ ROCm protection: ${maxTokens} tokens max for ${modelSize.toFixed(1)}GB model`);

      console.debug('Ollama chat request (safe mode):', {
        model: ollamaRequest.model,
        messageCount: ollamaRequest.messages.length,
        toolCount: tools.length,
        options: ollamaRequest.options
      });
      
      const response = await fetch(`${this.config.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
        signal: controller.signal, // Enable abort
      });

      clearTimeout(timeoutId); // Clear timeout on success

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API error ${response.status}: ${errorText}`);
      }

      const ollamaResponse: OllamaChatResponse = await response.json();
      console.debug('Ollama chat response received:', {
        hasMessage: !!ollamaResponse.message,
        hasContent: !!ollamaResponse.message?.content,
        hasToolCalls: !!ollamaResponse.message?.tool_calls,
        done: ollamaResponse.done
      });
      
      return this.ollamaChatToGeminiResponse(ollamaResponse, userMessage);
    } catch (error) {
      clearTimeout(timeoutId);
      
      const err = error as Error;
      this.detectGpuHang(err);
      
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
    return !!(request.config?.tools && Array.isArray(request.config.tools) && request.config.tools.length > 0);
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

    // Apply basic generation config (simplified for now)
    if (request.config) {
      ollamaRequest.options = {
        temperature: request.config.temperature,
        // Other options can be added later as needed
      };
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
    const hasTools = this.hasTools(request);
    const enableChatApi = this.config.enableChatApi ?? false;
    
    // INTELLIGENT MODEL ASSESSMENT: Check if model is suitable for Chat API
    const chatApiAssessment = this.shouldUseChatAPI(hasTools, enableChatApi);
    
    if (hasTools) {
      // For tool requests: Check if Chat API is safe to use
      console.debug('Ollama generateContentStream - tool request detected');
      
      if (!chatApiAssessment.shouldUse) {
        console.warn(`🛡️ Skipping Chat API: ${chatApiAssessment.reason}`);
        console.warn('Tools will not execute but system remains stable');
        return this.callGenerateAPIStream(request, userPromptId);
      }
      
      console.debug(`✅ Using Chat API: ${chatApiAssessment.reason}`);
      
      try {
        return this.callChatAPIStream(request, userPromptId);
      } catch (error) {
        this.detectGpuHang(error instanceof Error ? error : new Error(String(error)));
        console.warn('Chat API stream failed for tools:', error instanceof Error ? error.message : String(error));
        console.debug('Fallback to Generate API stream - tools will not be executed');
        return this.callGenerateAPIStream(request, userPromptId);
      }
    } else {
      // For non-tool requests: Generate API (stable and fast)
      console.debug('Ollama generateContentStream - using generate API for non-tool request');
      return this.callGenerateAPIStream(request, userPromptId);
    }
  }

  /**
   * Call Ollama Chat API Stream (for tool calling)
   */
  private async *callChatAPIStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const timeout = this.getChatTimeout() * 2; // Double timeout for streaming
    
    // Use AbortController for timeout and cleanup
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn(`Ollama chat stream timeout (${timeout}ms), aborting...`);
      controller.abort();
    }, timeout);

    try {
      // Simplified message conversion
      const userMessage = this.extractSimpleTextContent(request.contents);
      const tools = request.config?.tools ? this.convertGeminiToolsToOllama(request.config.tools) : [];
      
      const ollamaRequest: OllamaChatRequest = {
        model: this.config.model,
        messages: [{ role: 'user', content: userMessage }], // Keep it simple
        stream: true,
      };

      // Add tools if available
      if (tools.length > 0) {
        ollamaRequest.tools = tools;
      }

      // CRITICAL: Ultra-conservative streaming protection
      const modelSize = this.lastModelSize || 0;
      const isProblematicModel = modelSize > 8;
      
      // Even more conservative for streaming (streaming compounds VRAM issues)
      let maxTokens = 1024; // Half of non-streaming
      if (isProblematicModel) {
        maxTokens = this.gpuHangDetected ? 128 : 256; // Tiny responses for streaming
      }
      if (this.consecutiveTimeouts > 0) {
        maxTokens = Math.max(64, maxTokens / 2); // Minimal streaming after hangs
      }
      
      ollamaRequest.options = {
        temperature: request.config?.temperature || 0.7,
        num_predict: maxTokens,
        num_ctx: 1024, // Very small context for streaming stability
        repeat_penalty: 1.1,
      };
      
      console.debug(`🛡️ ROCm stream protection: ${maxTokens} tokens max for ${modelSize.toFixed(1)}GB model`);

      console.debug('Ollama chat stream request (safe mode):', {
        model: ollamaRequest.model,
        messageCount: ollamaRequest.messages.length,
        toolCount: tools.length
      });

      const response = await fetch(`${this.config.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ollamaRequest),
        signal: controller.signal, // Enable abort
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
                
                // Only yield if there's new content
                if (ollamaResponse.message && ollamaResponse.message.content) {
                  const geminiResponse = this.ollamaChatToGeminiResponse(ollamaResponse, userMessage);
                  
                  // For streaming, only include the new incremental text
                  if (geminiResponse.candidates && geminiResponse.candidates[0] && 
                      geminiResponse.candidates[0].content && geminiResponse.candidates[0].content.parts &&
                      geminiResponse.candidates[0].content.parts[0] && 
                      'text' in geminiResponse.candidates[0].content.parts[0]) {
                    // Only send the new incremental content
                    geminiResponse.candidates[0].content.parts[0].text = ollamaResponse.message.content;
                  }
                  
                  yield geminiResponse;
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
      this.detectGpuHang(err);
      
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

    // Apply basic generation config (simplified for now)
    if (request.config) {
      ollamaRequest.options = {
        temperature: request.config.temperature,
        // Other options can be added later as needed
      };
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