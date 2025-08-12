/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment, LoadedSettings } from './settings.js';
import { validateOllamaConfiguration } from './ollamaDiscovery.js';

export const validateAuthMethod = (authMethod: string): string | null => {
  loadEnvironment();
  if (
    authMethod === AuthType.LOGIN_WITH_GOOGLE ||
    authMethod === AuthType.CLOUD_SHELL
  ) {
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!process.env.GEMINI_API_KEY) {
      return 'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env.GOOGLE_CLOUD_PROJECT && !!process.env.GOOGLE_CLOUD_LOCATION;
    const hasGoogleApiKey = !!process.env.GOOGLE_API_KEY;
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'When using Vertex AI, you must specify either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_OLLAMA) {
    // For Ollama, we need to check if the service is available and has models
    // The actual validation will be done asynchronously during auth setup
    return null;
  }

  return 'Invalid auth method selected.';
};

/**
 * Async version of auth validation that can handle Ollama model discovery
 */
export const validateAuthMethodAsync = async (
  authMethod: string,
  settings?: LoadedSettings,
): Promise<string | null> => {
  // For non-Ollama methods, use sync validation
  if (authMethod !== AuthType.USE_OLLAMA) {
    return validateAuthMethod(authMethod);
  }

  // For Ollama, do deep validation including model checking
  if (!settings) {
    return 'Settings required for Ollama validation';
  }

  const validation = await validateOllamaConfiguration(settings);
  if (!validation.isValid) {
    return validation.message || 'Ollama configuration is invalid';
  }

  return null;
};
