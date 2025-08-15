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
    
    // Save model, base URL, and default configuration to settings
    settings.setValue(SettingScope.User, 'ollamaModel', modelName);
    settings.setValue(SettingScope.User, 'ollamaBaseUrl', ollamaBaseUrl);
    
    // Set defaults with comments for configuration values
    if (!settings.merged.ollamaChatTimeout) {
      settings.setValue(SettingScope.User, 'ollamaChatTimeout', 120); // Default: 2 minutes
    }
    if (!settings.merged.ollamaStreamingTimeout) {
      settings.setValue(SettingScope.User, 'ollamaStreamingTimeout', 300); // Default: 5 minutes for streaming
    }
    if (!settings.merged.ollamaContextLimit) {
      settings.setValue(SettingScope.User, 'ollamaContextLimit', 2048); // Default: Conservative 2K context
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