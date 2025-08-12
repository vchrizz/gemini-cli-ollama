/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// Global cache for Ollama model context lengths
const ollamaContextLengthCache = new Map<string, number>();

/**
 * Set the context length for an Ollama model
 */
export function setOllamaModelContextLength(modelName: string, contextLength: number): void {
  ollamaContextLengthCache.set(modelName, contextLength);
}

/**
 * Get the context length for an Ollama model
 */
export function getOllamaModelContextLength(modelName: string): number | undefined {
  return ollamaContextLengthCache.get(modelName);
}

/**
 * Check if a model name appears to be an Ollama model
 * (not starting with "gemini-")
 */
export function isOllamaModel(modelName: string): boolean {
  return !modelName.startsWith('gemini-') && !modelName.startsWith('models/gemini-');
}

/**
 * Clear the cache (useful for testing)
 */
export function clearOllamaContextLengthCache(): void {
  ollamaContextLengthCache.clear();
}

/**
 * Fetch context length from Ollama API and cache it
 */
export async function fetchAndCacheOllamaContextLength(
  modelName: string, 
  baseUrl: string = 'http://localhost:11434'
): Promise<number> {
  try {
    const response = await fetch(`${baseUrl}/api/show`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: modelName }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.status} ${response.statusText}`);
    }
    
    const modelInfo = await response.json();
    
    // Look for context_length in model_info
    for (const [key, value] of Object.entries(modelInfo.model_info || {})) {
      if (key.endsWith('context_length')) {
        const contextLength = typeof value === 'number' ? value : parseInt(value as string, 10);
        setOllamaModelContextLength(modelName, contextLength);
        return contextLength;
      }
    }
    
    // Default context length if not found
    const defaultContextLength = 2048;
    setOllamaModelContextLength(modelName, defaultContextLength);
    return defaultContextLength;
  } catch (error) {
    console.warn(`Failed to fetch context length for ${modelName}:`, error);
    // Set and return default value
    const defaultContextLength = 2048;
    setOllamaModelContextLength(modelName, defaultContextLength);
    return defaultContextLength;
  }
}