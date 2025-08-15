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
  contextLimit?: number; // Context window size (num_ctx)
  debugLogging?: boolean; // Enable debug logging to file
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

  constructor(config: OllamaConfig) {
    this.config = config;
    
    // Synchronously start context length initialization but don't wait for it
    this.initializeContextLength();
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
    
    // Extract user message - simplified to match working cURL
    let userContent = '';
    
    // Extract user content from request
    if (typeof request.contents === 'string') {
      userContent = request.contents;
    } else if (Array.isArray(request.contents)) {
      // Extract the LAST user message from conversation history
      const contents = request.contents as Content[];
      for (let i = contents.length - 1; i >= 0; i--) {
        const content = contents[i];
        if (content.role === 'user' && content.parts) {
          const text = content.parts
            .map((part) => {
              if (typeof part === 'string') return part;
              if (typeof part === 'object' && 'text' in part) return part.text;
              return '';
            })
            .join('')
            .trim();
          if (text) {
            userContent = text;
            break;
          }
        }
      }
    } else {
      // For any other complex content, try extract text parts
      userContent = this.extractTextFromContent(request.contents as any);
    }
    
    // Optional debug logging to file (if enabled in config)
    if (this.config.debugLogging) {
      const debugLog = {
        timestamp: new Date().toISOString(),
        contentType: typeof request.contents,
        extractedContent: userContent,
        hasTools: this.hasTools(request),
        requestSummary: {
          isString: typeof request.contents === 'string',
          isArray: Array.isArray(request.contents),
          arrayLength: Array.isArray(request.contents) ? request.contents.length : 0
        }
      };
      
      import('fs').then(fs => {
        fs.appendFileSync('./debug.log', 
          JSON.stringify(debugLog, null, 2) + '\n---\n');
      }).catch(err => console.warn('Debug log failed:', err));
    }
    
    // Always add user message
    if (!userContent) {
      console.warn('⚠️ No user content found in request, this should not happen!');
      console.log('Request contents:', JSON.stringify(request.contents, null, 2));
    }
    
    messages.push({
      role: 'user',
      content: userContent || "Please provide a valid command to execute."
    });
    
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
    const hasTools = this.hasTools(request);
    const enableChatApi = this.config.enableChatApi ?? true; // Default to Chat API for tool calling
    
    const tools = (request as any).tools || request.config?.tools;
    console.log('🚀 Ollama generateContent called with:', {
      hasTools,
      enableChatApi,
      toolsLength: tools ? Array.isArray(tools) ? tools.length : 'not array' : 'no tools',
      contentType: typeof request.contents,
      willUseChatAPI: hasTools && this.shouldUseChatAPI(hasTools, enableChatApi)
    });
    
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

      // Use configurable options to prevent GPU hang
      ollamaRequest.options = {
        temperature: request.config?.temperature || 0.7,
        num_ctx: this.config.contextLimit || 2048, // Use configured context limit
      };

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

    // Apply basic generation config (conservative to prevent GPU hang)
    if (request.config) {
      ollamaRequest.options = {
        temperature: request.config.temperature,
        num_ctx: this.config.contextLimit || 2048, // Use configured context limit
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
    const enableChatApi = this.config.enableChatApi ?? true;
    
    if (hasTools && this.shouldUseChatAPI(hasTools, enableChatApi)) {
      // Use Chat API for tool calling
      console.debug('Ollama generateContentStream - using Chat API for tool request');
      try {
        return this.callChatAPIStream(request, userPromptId);
      } catch (error) {
        console.warn('Chat API stream failed for tools:', error instanceof Error ? error.message : String(error));
        console.debug('Fallback to Generate API stream - tools will not be executed');
        return this.callGenerateAPIStream(request, userPromptId);
      }
    } else {
      // Use Generate API for non-tool requests or when Chat API is disabled
      console.debug('Ollama generateContentStream - using Generate API');
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
    const timeout = this.getTimeout() * 2; // Double timeout for streaming
    
    // Use AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn(`Ollama chat stream timeout (${timeout}ms), aborting...`);
      controller.abort();
    }, timeout);

    try {
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

      // Use configurable options to prevent GPU hang
      ollamaRequest.options = {
        temperature: request.config?.temperature || 0.7,
        num_ctx: this.config.contextLimit || 2048, // Use configured context limit
      };

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

    // Apply basic generation config (conservative to prevent GPU hang)
    if (request.config) {
      ollamaRequest.options = {
        temperature: request.config.temperature,
        num_ctx: this.config.contextLimit || 2048, // Use configured context limit
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