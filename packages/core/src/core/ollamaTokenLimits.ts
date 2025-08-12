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