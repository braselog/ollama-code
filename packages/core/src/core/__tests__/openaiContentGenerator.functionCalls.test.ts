/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenAIContentGenerator } from '../openaiContentGenerator.js';
import { Config } from '../../config/config.js';
import type { ChatCompletion, ChatCompletionChunk } from 'openai/resources/chat/index.js';

describe('OpenAIContentGenerator - Function Calls', () => {
  let contentGenerator: OpenAIContentGenerator;
  let mockConfig: Config;

  beforeEach(() => {
    mockConfig = {
      getContentGeneratorConfig: vi.fn(() => ({
        model: 'qwen3:14b',
        authType: 'openai',
        apiKey: 'test-key',
      })),
    } as unknown as Config;

    contentGenerator = new OpenAIContentGenerator('test-key', 'qwen3:14b', mockConfig);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('convertToGeminiFormat', () => {
    it('should populate functionCalls property when tool_calls are present', () => {
      const mockOpenAIResponse: ChatCompletion = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'qwen3:14b',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'I will help you with that.',
              tool_calls: [
                {
                  id: 'call_123',
                  type: 'function',
                  function: {
                    name: 'web_fetch',
                    arguments: '{"prompt":"Summarize the purpose of @OLLAMA.md"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
        },
      };

      // Access the private method via type assertion
      const convertMethod = (contentGenerator as any).convertToGeminiFormat.bind(contentGenerator);
      const geminiResponse = convertMethod(mockOpenAIResponse);

      // Verify functionCalls property is populated
      expect(geminiResponse.functionCalls).toBeDefined();
      expect(geminiResponse.functionCalls).toHaveLength(1);
      expect(geminiResponse.functionCalls[0]).toEqual({
        id: 'call_123',
        name: 'web_fetch',
        args: { prompt: 'Summarize the purpose of @OLLAMA.md' },
      });

      // Verify parts array also contains functionCall
      expect(geminiResponse.candidates).toBeDefined();
      expect(geminiResponse.candidates[0].content.parts).toBeDefined();
      const functionCallParts = geminiResponse.candidates[0].content.parts.filter(
        (part: any) => part.functionCall
      );
      expect(functionCallParts).toHaveLength(1);
      expect(functionCallParts[0].functionCall).toEqual({
        id: 'call_123',
        name: 'web_fetch',
        args: { prompt: 'Summarize the purpose of @OLLAMA.md' },
      });
    });

    it('should handle multiple tool calls', () => {
      const mockOpenAIResponse: ChatCompletion = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'qwen3:14b',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'tool_one',
                    arguments: '{"arg1":"value1"}',
                  },
                },
                {
                  id: 'call_2',
                  type: 'function',
                  function: {
                    name: 'tool_two',
                    arguments: '{"arg2":"value2"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
            logprobs: null,
          },
        ],
      };

      const convertMethod = (contentGenerator as any).convertToGeminiFormat.bind(contentGenerator);
      const geminiResponse = convertMethod(mockOpenAIResponse);

      expect(geminiResponse.functionCalls).toHaveLength(2);
      expect(geminiResponse.functionCalls[0].name).toBe('tool_one');
      expect(geminiResponse.functionCalls[1].name).toBe('tool_two');
    });

    it('should not populate functionCalls when no tool_calls present', () => {
      const mockOpenAIResponse: ChatCompletion = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'qwen3:14b',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'This is just a text response.',
            },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
      };

      const convertMethod = (contentGenerator as any).convertToGeminiFormat.bind(contentGenerator);
      const geminiResponse = convertMethod(mockOpenAIResponse);

      expect(geminiResponse.functionCalls).toBeUndefined();
    });
  });

  describe('convertStreamChunkToGeminiFormat', () => {
    it('should populate functionCalls property in final streaming chunk', () => {
      // Simulate streaming chunks
      const chunks: ChatCompletionChunk[] = [
        {
          id: 'test-id',
          object: 'chat.completion.chunk',
          created: Date.now(),
          model: 'qwen3:14b',
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_abc',
                    type: 'function',
                    function: {
                      name: 'web_fetch',
                      arguments: '',
                    },
                  },
                ],
              },
              finish_reason: null,
              logprobs: null,
            },
          ],
        },
        {
          id: 'test-id',
          object: 'chat.completion.chunk',
          created: Date.now(),
          model: 'qwen3:14b',
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: {
                      arguments: '{"url":"',
                    },
                  },
                ],
              },
              finish_reason: null,
              logprobs: null,
            },
          ],
        },
        {
          id: 'test-id',
          object: 'chat.completion.chunk',
          created: Date.now(),
          model: 'qwen3:14b',
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: {
                      arguments: 'example.com"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
              logprobs: null,
            },
          ],
        },
      ];

      const convertMethod = (contentGenerator as any).convertStreamChunkToGeminiFormat.bind(contentGenerator);
      
      // Process all chunks
      let finalResponse;
      for (const chunk of chunks) {
        finalResponse = convertMethod(chunk);
      }

      // Only the final chunk should have functionCalls populated
      expect(finalResponse.functionCalls).toBeDefined();
      expect(finalResponse.functionCalls).toHaveLength(1);
      expect(finalResponse.functionCalls[0]).toEqual({
        id: 'call_abc',
        name: 'web_fetch',
        args: { url: 'example.com' },
      });
    });

    it('should not populate functionCalls in intermediate streaming chunks', () => {
      const chunk: ChatCompletionChunk = {
        id: 'test-id',
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: 'qwen3:14b',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_xyz',
                  type: 'function',
                  function: {
                    name: 'some_tool',
                    arguments: '{"partial":',
                  },
                },
              ],
            },
            finish_reason: null, // Not finished yet
            logprobs: null,
          },
        ],
      };

      const convertMethod = (contentGenerator as any).convertStreamChunkToGeminiFormat.bind(contentGenerator);
      const geminiResponse = convertMethod(chunk);

      // Should not have functionCalls yet since streaming is not complete
      expect(geminiResponse.functionCalls).toBeUndefined();
    });
  });

  describe('combineStreamResponsesForLogging', () => {
    it('should populate functionCalls in combined response', () => {
      // Create mock streaming responses that would have been generated
      const mockResponses = [
        {
          candidates: [
            {
              content: {
                parts: [{ text: 'Let me help you with that.' }],
                role: 'model' as const,
              },
              finishReason: 'STOP' as const,
              index: 0,
              safetyRatings: [],
            },
          ],
          modelVersion: 'qwen3:14b',
          promptFeedback: { safetyRatings: [] },
        },
        {
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      id: 'call_final',
                      name: 'search_tool',
                      args: { query: 'test query' },
                    },
                  },
                ],
                role: 'model' as const,
              },
              finishReason: 'STOP' as const,
              index: 0,
              safetyRatings: [],
            },
          ],
          modelVersion: 'qwen3:14b',
          promptFeedback: { safetyRatings: [] },
          functionCalls: [
            {
              id: 'call_final',
              name: 'search_tool',
              args: { query: 'test query' },
            },
          ],
        },
      ];

      const combineMethod = (contentGenerator as any).combineStreamResponsesForLogging.bind(contentGenerator);
      const combinedResponse = combineMethod(mockResponses);

      expect(combinedResponse.functionCalls).toBeDefined();
      expect(combinedResponse.functionCalls).toHaveLength(1);
      expect(combinedResponse.functionCalls[0]).toEqual({
        id: 'call_final',
        name: 'search_tool',
        args: { query: 'test query' },
      });
    });
  });
});
