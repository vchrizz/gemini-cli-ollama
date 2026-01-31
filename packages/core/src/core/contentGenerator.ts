/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import { GoogleGenAI } from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import type { Config } from '../config/config.js';
import { loadApiKey } from './apiKeyCredentialStorage.js';

import type { UserTierId } from '../code_assist/types.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { InstallationManager } from '../utils/installationManager.js';
import { FakeContentGenerator } from './fakeContentGenerator.js';
import { parseCustomHeaders } from '../utils/customHeaderUtils.js';
import { RecordingContentGenerator } from './recordingContentGenerator.js';
import { OllamaContentGenerator } from './ollamaContentGenerator.js';
import { getVersion, resolveModel } from '../../index.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  userTier?: UserTierId;

  userTierName?: string;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  LEGACY_CLOUD_SHELL = 'cloud-shell',
  COMPUTE_ADC = 'compute-default-credentials',
  USE_OLLAMA = 'ollama',
}

export type ContentGeneratorConfig = {
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType;
  proxy?: string;
  model?: string;
  ollamaBaseUrl?: string;
  ollamaEnableChatApi?: boolean;
  ollamaChatTimeout?: number;
  ollamaStreamingTimeout?: number;
  ollamaContextLimit?: number;
  ollamaRequestContextSize?: number;
  ollamaTemperature?: number;
  ollamaDebugLogging?: boolean;
};

export async function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
): Promise<ContentGeneratorConfig> {
  const geminiApiKey =
    process.env['GEMINI_API_KEY'] || (await loadApiKey()) || undefined;
  const googleApiKey = process.env['GOOGLE_API_KEY'] || undefined;
  const googleCloudProject =
    process.env['GOOGLE_CLOUD_PROJECT'] ||
    process.env['GOOGLE_CLOUD_PROJECT_ID'] ||
    undefined;
  const googleCloudLocation = process.env['GOOGLE_CLOUD_LOCATION'] || undefined;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    authType,
    proxy: config?.getProxy(),
    model: config?.getModel(),
    ollamaBaseUrl: config?.getOllamaBaseUrl(),
    ollamaEnableChatApi: config?.getOllamaEnableChatApi(),
    ollamaChatTimeout: config?.getOllamaChatTimeout(),
    ollamaStreamingTimeout: config?.getOllamaStreamingTimeout(),
    ollamaContextLimit: config?.getOllamaContextLimit(),
    ollamaRequestContextSize: config?.getOllamaRequestContextSize(),
    ollamaTemperature: config?.getOllamaTemperature(),
    ollamaDebugLogging: config?.getOllamaDebugLogging(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.COMPUTE_ADC
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_OLLAMA) {
    // For Ollama, we just need to ensure the base URL is set
    // Model validation will be done when creating the content generator
    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const generator = await (async () => {
    if (gcConfig.fakeResponses) {
      const fakeGenerator = await FakeContentGenerator.fromFile(
        gcConfig.fakeResponses,
      );
      return new LoggingContentGenerator(fakeGenerator, gcConfig);
    }
    const version = await getVersion();
    const model = resolveModel(
      gcConfig.getModel(),
      gcConfig.getPreviewFeatures(),
    );
    const customHeadersEnv =
      process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
    const userAgent = `GeminiCLI/${version}/${model} (${process.platform}; ${process.arch})`;
    const customHeadersMap = parseCustomHeaders(customHeadersEnv);
    const apiKeyAuthMechanism =
      process.env['GEMINI_API_KEY_AUTH_MECHANISM'] || 'x-goog-api-key';
    const apiVersionEnv = process.env['GOOGLE_GENAI_API_VERSION'];

    const baseHeaders: Record<string, string> = {
      ...customHeadersMap,
      'User-Agent': userAgent,
    };

    if (
      apiKeyAuthMechanism === 'bearer' &&
      (config.authType === AuthType.USE_GEMINI ||
        config.authType === AuthType.USE_VERTEX_AI) &&
      config.apiKey
    ) {
      baseHeaders['Authorization'] = `Bearer ${config.apiKey}`;
    }
    if (
      config.authType === AuthType.LOGIN_WITH_GOOGLE ||
      config.authType === AuthType.COMPUTE_ADC
    ) {
      const httpOptions = { headers: baseHeaders };
      return new LoggingContentGenerator(
        await createCodeAssistContentGenerator(
          httpOptions,
          config.authType,
          gcConfig,
          sessionId,
        ),
        gcConfig,
      );
    }

    if (
      config.authType === AuthType.USE_GEMINI ||
      config.authType === AuthType.USE_VERTEX_AI
    ) {
      let headers: Record<string, string> = { ...baseHeaders };
      if (gcConfig?.getUsageStatisticsEnabled()) {
        const installationManager = new InstallationManager();
        const installationId = installationManager.getInstallationId();
        headers = {
          ...headers,
          'x-gemini-api-privileged-user-id': `${installationId}`,
        };
      }
      const httpOptions = { headers };

      const googleGenAI = new GoogleGenAI({
        apiKey: config.apiKey === '' ? undefined : config.apiKey,
        vertexai: config.vertexai,
        httpOptions,
        ...(apiVersionEnv && { apiVersion: apiVersionEnv }),
      });
      return new LoggingContentGenerator(googleGenAI.models, gcConfig);
    }
    if (config.authType === AuthType.USE_OLLAMA) {
      const debugLoggingEnabled = config.ollamaDebugLogging || false;

      if (debugLoggingEnabled) {
        console.log('Creating OllamaContentGenerator with config:', {
          authType: config.authType,
          baseUrl: config.ollamaBaseUrl || 'http://localhost:11434',
          model: config.model,
          enableChatApi: config.ollamaEnableChatApi,
          debugLogging: debugLoggingEnabled
        });
      }

      const ollamaConfig = {
        baseUrl: config.ollamaBaseUrl || 'http://localhost:11434',
        model: config.model,
        enableChatApi: config.ollamaEnableChatApi,
        timeout: (config.ollamaChatTimeout || 300) * 1000,
        streamingTimeout: (config.ollamaStreamingTimeout || 600) * 1000,
        contextLimit: config.ollamaContextLimit || 8192,
        requestContextSize: config.ollamaRequestContextSize || 8192,
        temperature: config.ollamaTemperature || 0.7,
        debugLogging: debugLoggingEnabled,
      };
      const ollamaGenerator = new OllamaContentGenerator(ollamaConfig);

      await ollamaGenerator.ensureContextLengthInitialized();

      if (debugLoggingEnabled) {
        console.log('OllamaContentGenerator created and initialized');
      }
      return new LoggingContentGenerator(ollamaGenerator, gcConfig);
    }

    throw new Error(
      `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
    );
  })();

  if (gcConfig.recordResponses) {
    return new RecordingContentGenerator(generator, gcConfig.recordResponses);
  }

  return generator;
}
