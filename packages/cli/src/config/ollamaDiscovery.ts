/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { LoadedSettings, SettingScope } from './settings.js';

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

interface OllamaListResponse {
  models: OllamaModel[];
}

/**
 * Check if Ollama is running and accessible
 */
export async function checkOllamaConnection(baseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });
    return response.ok;
  } catch (error) {
    return false;
  }
}

/**
 * Get available models from Ollama
 */
export async function getOllamaModels(baseUrl: string): Promise<OllamaModel[]> {
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
    }
    
    const data: OllamaListResponse = await response.json();
    return data.models || [];
  } catch (error) {
    throw new Error(`Failed to get Ollama models: ${error}`);
  }
}

/**
 * Get model information including context length
 */
export async function getOllamaModelInfo(baseUrl: string, modelName: string): Promise<OllamaShowResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/show`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: modelName }),
      signal: AbortSignal.timeout(15000), // 15 second timeout for model info
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
export function extractContextLength(modelInfo: OllamaShowResponse): number {
  try {
    // Look for context_length in model_info
    for (const [key, value] of Object.entries(modelInfo.model_info || {})) {
      if (key.endsWith('context_length')) {
        const contextLength = typeof value === 'number' ? value : parseInt(value as string, 10);
        if (!isNaN(contextLength) && contextLength > 0) {
          return contextLength;
        }
      }
    }
    
    // Parse modelfile for num_ctx parameter
    if (modelInfo.modelfile) {
      const numCtxMatch = modelInfo.modelfile.match(/PARAMETER\s+num_ctx\s+(\d+)/i);
      if (numCtxMatch) {
        const contextLength = parseInt(numCtxMatch[1], 10);
        if (!isNaN(contextLength) && contextLength > 0) {
          return contextLength;
        }
      }
    }
    
    // Fallback based on model size/type
    const parameterSize = modelInfo.details?.parameter_size || '';
    if (parameterSize.includes('70b') || parameterSize.includes('72b')) {
      return 32768; // 32K context for large models
    } else if (parameterSize.includes('13b') || parameterSize.includes('34b')) {
      return 16384; // 16K context for medium models
    } else if (parameterSize.includes('7b') || parameterSize.includes('8b')) {
      return 8192; // 8K context for 7B models
    } else {
      return 4096; // 4K context as fallback
    }
  } catch (error) {
    console.warn(`Failed to extract context length for model:`, error);
    return 4096; // Safe fallback
  }
}

/**
 * Discover and configure an Ollama model if none is set
 */
export async function discoverAndConfigureOllamaModel(
  settings: LoadedSettings,
): Promise<{ success: boolean; model?: string; error?: string }> {
  const ollamaBaseUrl = settings.merged.ollamaBaseUrl || 'http://localhost:11434';
  
  // Check if Ollama is running
  const isConnected = await checkOllamaConnection(ollamaBaseUrl);
  if (!isConnected) {
    return {
      success: false,
      error: `Cannot connect to Ollama at ${ollamaBaseUrl}. Please ensure Ollama is running and accessible.`,
    };
  }
  
  try {
    // Get available models
    const models = await getOllamaModels(ollamaBaseUrl);
    
    if (models.length === 0) {
      return {
        success: false,
        error: `No models found in Ollama. Please install a model first using: ollama pull <model-name>`,
      };
    }
    
    // Select the first available model as default
    const firstModel = models[0];
    const modelName = firstModel.name;
    
    // Get model info to determine optimal context size
    let detectedContextLimit = 8192; // Default fallback
    try {
      console.log(`🔍 Detecting context limit for model: ${modelName}...`);
      const modelInfo = await getOllamaModelInfo(ollamaBaseUrl, modelName);
      detectedContextLimit = extractContextLength(modelInfo);
      console.log(`✅ Detected context limit: ${detectedContextLimit} for model ${modelName}`);
    } catch (error) {
      console.warn(`⚠️ Failed to detect context limit for ${modelName}, using default: ${detectedContextLimit}`, error);
    }
    
    // Save model, base URL, and detected configuration to settings
    settings.setValue(SettingScope.User, 'ollamaModel', modelName);
    settings.setValue(SettingScope.User, 'ollamaBaseUrl', ollamaBaseUrl);
    
    // Set optimized defaults based on model capabilities
    const timeoutForModel = detectedContextLimit >= 16384 ? 300 : 180; // Larger models need more timeout
    const streamingTimeoutForModel = detectedContextLimit >= 16384 ? 600 : 400;
    
    if (!settings.merged.ollamaChatTimeout) {
      settings.setValue(SettingScope.User, 'ollamaChatTimeout', timeoutForModel); // Adaptive timeout based on model size
    }
    if (!settings.merged.ollamaStreamingTimeout) {
      settings.setValue(SettingScope.User, 'ollamaStreamingTimeout', streamingTimeoutForModel); // Adaptive streaming timeout
    }
    if (!settings.merged.ollamaContextLimit) {
      // Use 75% of detected context for conversation tracking
      const optimalContext = Math.floor(detectedContextLimit * 0.75);
      settings.setValue(SettingScope.User, 'ollamaContextLimit', optimalContext);
      console.log(`📝 Set conversation context limit to ${optimalContext} (75% of ${detectedContextLimit}) for history management`);
    }
    if (!settings.merged.ollamaRequestContextSize) {
      // Default to 8192 for per-request context (matches typical Ollama server config)
      const defaultRequestContext = 8192;
      settings.setValue(SettingScope.User, 'ollamaRequestContextSize', defaultRequestContext);
      console.log(`📝 Set request context size to ${defaultRequestContext} (should match Ollama server --ctx-size)`);
    }
    if (!settings.merged.ollamaTemperature) {
      // Default balanced creativity
      settings.setValue(SettingScope.User, 'ollamaTemperature', 0.7);
      console.log(`📝 Set temperature to 0.7 (balanced creativity)`);
    }
    if (settings.merged.ollamaEnableChatApi === undefined) {
      settings.setValue(SettingScope.User, 'ollamaEnableChatApi', true); // Default: Enable Chat API for tool calling
    }
    if (settings.merged.ollamaDebugLogging === undefined) {
      settings.setValue(SettingScope.User, 'ollamaDebugLogging', false); // Default: Debug logging disabled for performance
    }
    
    return {
      success: true,
      model: modelName,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Update context limit for existing model configuration
 */
export async function updateContextLimitForModel(
  settings: LoadedSettings,
  modelName: string,
): Promise<{ success: boolean; newContextLimit?: number; error?: string }> {
  const ollamaBaseUrl = settings.merged.ollamaBaseUrl || 'http://localhost:11434';
  
  try {
    console.log(`🔍 Updating context limit for existing model: ${modelName}...`);
    const modelInfo = await getOllamaModelInfo(ollamaBaseUrl, modelName);
    const detectedContextLimit = extractContextLength(modelInfo);
    
    // Use 75% of detected context to leave room for response
    const optimalContext = Math.floor(detectedContextLimit * 0.75);
    const currentContext = settings.merged.ollamaContextLimit || 2048;
    
    if (optimalContext !== currentContext) {
      settings.setValue(SettingScope.User, 'ollamaContextLimit', optimalContext);
      console.log(`📝 Updated context limit from ${currentContext} to ${optimalContext} for model ${modelName}`);
      
      return {
        success: true,
        newContextLimit: optimalContext,
      };
    } else {
      return {
        success: true,
        newContextLimit: currentContext,
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Validate Ollama configuration and suggest fixes
 */
export async function validateOllamaConfiguration(
  settings: LoadedSettings,
): Promise<{ isValid: boolean; message?: string; suggestedModel?: string }> {
  const ollamaBaseUrl = settings.merged.ollamaBaseUrl || 'http://localhost:11434';
  const ollamaModel = settings.merged.ollamaModel;
  
  // Check connection
  const isConnected = await checkOllamaConnection(ollamaBaseUrl);
  if (!isConnected) {
    return {
      isValid: false,
      message: `Cannot connect to Ollama at ${ollamaBaseUrl}. Please ensure Ollama is running.`,
    };
  }
  
  // If no model is configured, try to suggest one
  if (!ollamaModel) {
    try {
      const models = await getOllamaModels(ollamaBaseUrl);
      if (models.length === 0) {
        return {
          isValid: false,
          message: `No models found in Ollama. Install a model first using: ollama pull llama2`,
        };
      }
      
      return {
        isValid: false,
        message: `No Ollama model configured. Available models: ${models.map(m => m.name).join(', ')}`,
        suggestedModel: models[0].name,
      };
    } catch (error) {
      return {
        isValid: false,
        message: `Failed to fetch available models: ${error}`,
      };
    }
  }
  
  // Check if the configured model exists
  try {
    const models = await getOllamaModels(ollamaBaseUrl);
    const modelExists = models.some(m => m.name === ollamaModel);
    
    if (!modelExists) {
      return {
        isValid: false,
        message: `Model "${ollamaModel}" not found in Ollama. Available models: ${models.map(m => m.name).join(', ')}`,
        suggestedModel: models.length > 0 ? models[0].name : undefined,
      };
    }
    
    return { isValid: true };
  } catch (error) {
    return {
      isValid: false,
      message: `Failed to validate model configuration: ${error}`,
    };
  }
}