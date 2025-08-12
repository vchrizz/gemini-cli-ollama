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
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { setOllamaModelContextLength } from './ollamaTokenLimits.js';

interface OllamaConfig {
  baseUrl: string;
  model: string;
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
  };
}

interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
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
    // Asynchronously discover and cache the context length
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
   * Convert Gemini-style contents to Ollama prompt
   */
  private contentsToPrompt(contents: ContentListUnion): string {
    // Handle different content union types
    if (typeof contents === 'string') {
      return contents;
    }
    
    if (Array.isArray(contents)) {
      if (contents.length === 0) return '';
      
      // Check if it's an array of Content objects
      if (typeof contents[0] === 'object' && 'parts' in contents[0]) {
        return (contents as Content[])
          .map((content) => {
            if (content.parts) {
              return content.parts
                .map((part) => {
                  if (typeof part === 'string') {
                    return part;
                  }
                  if (typeof part === 'object' && 'text' in part) {
                    return part.text;
                  }
                  // For now, skip non-text parts (images, etc.)
                  return '';
                })
                .join('');
            }
            return '';
          })
          .join('\n');
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
   * Convert Ollama response to Gemini-style response
   */
  private ollamaToGeminiResponse(
    ollamaResponse: OllamaGenerateResponse,
  ): GenerateContentResponse {
    const usageMetadata = new GenerateContentResponseUsageMetadata();
    usageMetadata.promptTokenCount = ollamaResponse.prompt_eval_count || 0;
    usageMetadata.candidatesTokenCount = ollamaResponse.eval_count || 0;

    const candidate: Candidate = {
      content: {
        parts: [{ text: ollamaResponse.response }],
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

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    let prompt = this.contentsToPrompt(request.contents);
    
    const ollamaRequest: OllamaGenerateRequest = {
      model: this.config.model,
      prompt: prompt,
      stream: false,
    };

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
      
      // Check if we got an empty response
      if (!ollamaResponse.response || ollamaResponse.response.trim() === '') {
        throw new Error('Ollama returned an empty response. This may indicate the model cannot handle the requested JSON schema or the prompt is too complex.');
      }
      
      return this.ollamaToGeminiResponse(ollamaResponse);
    } catch (error) {
      throw new Error(`Failed to generate content with Ollama: ${error}`);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.generateStreamInternal(request, userPromptId);
  }

  private async *generateStreamInternal(
    request: GenerateContentParameters,
    userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse> {
    let prompt = this.contentsToPrompt(request.contents);
    
    const ollamaRequest: OllamaGenerateRequest = {
      model: this.config.model,
      prompt: prompt,
      stream: true,
    };

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
                  
                  const geminiResponse = this.ollamaToGeminiResponse(ollamaResponse);
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