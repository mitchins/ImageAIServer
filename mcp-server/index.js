#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import fetch from "node-fetch";

const server = new Server(
  {
    name: "imageai-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Default server configuration
const DEFAULT_SERVER_URL = "http://localhost:8001";
const SERVER_URL = process.env.IMAGEAI_SERVER_URL || DEFAULT_SERVER_URL;

// Helper function to make API requests
async function makeImageRequest(prompt, options = {}) {
  const requestBody = {
    prompt: prompt,
    model: options.model || "sd15-onnx",
    n: options.n || 1,
    width: options.width || 512,
    height: options.height || 512,
    negative_prompt: options.negative_prompt || "",
  };

  try {
    const response = await fetch(`${SERVER_URL}/v1/images/generations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`Failed to generate image: ${error.message}`);
  }
}

// Helper function to get available models
async function getAvailableModels() {
  try {
    const response = await fetch(`${SERVER_URL}/v1/models/generation`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const models = await response.json();
    return models;
  } catch (error) {
    throw new Error(`Failed to get models: ${error.message}`);
  }
}

// Helper function to recommend the best model based on requirements
function recommendModel(prompt, options = {}) {
  const { requiresHighQuality, requiresSpeed, hasGPU, memoryConstrained } = options;
  
  // Model performance characteristics
  const modelRecommendations = {
    "sd15-onnx": {
      quality: 3,
      speed: 4,
      memoryUsage: 1,
      cpuCompatible: true,
      description: "Best for CPU-only systems, low memory, or quick testing"
    },
    "sdxl-turbo": {
      quality: 4,
      speed: 5,
      memoryUsage: 4,
      cpuCompatible: false,
      description: "Best for fast, good quality results with GPU"
    },
    "sdxl": {
      quality: 5,
      speed: 2,
      memoryUsage: 4,
      cpuCompatible: false,
      description: "Best overall quality, slower generation, requires GPU"
    },
    "flux1-schnell": {
      quality: 5,
      speed: 3,
      memoryUsage: 5,
      cpuCompatible: false,
      description: "Excellent quality, good for complex prompts, requires powerful GPU"
    }
  };

  let recommendations = [];
  
  if (memoryConstrained) {
    recommendations.push({
      model: "sd15-onnx",
      reason: "Low memory usage (~500MB), works on any system"
    });
  } else if (requiresHighQuality && hasGPU) {
    recommendations.push({
      model: "sdxl",
      reason: "Highest quality output, best for detailed images"
    });
    recommendations.push({
      model: "flux1-schnell", 
      reason: "Excellent for complex prompts and artistic images"
    });
  } else if (requiresSpeed && hasGPU) {
    recommendations.push({
      model: "sdxl-turbo",
      reason: "Fast generation with good quality"
    });
  } else if (!hasGPU) {
    recommendations.push({
      model: "sd15-onnx",
      reason: "Only CPU-compatible option available"
    });
  } else {
    // Default balanced recommendation
    recommendations.push({
      model: "sdxl-turbo",
      reason: "Good balance of speed and quality"
    });
  }
  
  return recommendations;
}

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "generate_image",
        description: "Generate an image using AI models. For best results: use SDXL for high quality, SDXL-Turbo for speed, SD1.5-ONNX for CPU/low memory systems, FLUX for complex artistic prompts.",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description: "Text description of the image to generate",
            },
            model: {
              type: "string",
              description: "Model to use:\n- sd15-onnx: CPU compatible, low memory (~500MB), good for testing\n- sdxl: Best quality, requires GPU (~8GB VRAM)\n- sdxl-turbo: Fast generation, good quality, requires GPU (~8GB VRAM)\n- flux1-schnell: Excellent for complex/artistic prompts, requires powerful GPU\n- qwen-image: Alternative model option",
              default: "sd15-onnx",
            },
            width: {
              type: "number",
              description: "Image width in pixels (SD1.5: 256-768, SDXL: 512-1024)",
              default: 512,
            },
            height: {
              type: "number",
              description: "Image height in pixels (SD1.5: 256-768, SDXL: 512-1024)", 
              default: 512,
            },
            negative_prompt: {
              type: "string",
              description: "Text describing what to avoid in the image (not supported by FLUX)",
              default: "",
            },
            n: {
              type: "number",
              description: "Number of images to generate",
              default: 1,
              minimum: 1,
              maximum: 4,
            },
          },
          required: ["prompt"],
        },
      },
      {
        name: "recommend_model",
        description: "Get model recommendations based on your requirements (quality, speed, hardware constraints)",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description: "The image prompt you want to generate",
            },
            priority: {
              type: "string",
              enum: ["quality", "speed", "memory"],
              description: "What's most important: quality (best results), speed (fast generation), or memory (low resource usage)",
            },
            hardware: {
              type: "string",
              enum: ["cpu_only", "gpu_available", "powerful_gpu"],
              description: "Available hardware: cpu_only (no GPU), gpu_available (~8GB VRAM), powerful_gpu (16GB+ VRAM)",
              default: "cpu_only",
            },
          },
          required: ["prompt"],
        },
      },
      {
        name: "list_models",
        description: "List available image generation models and their capabilities",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "generate_image": {
        const { prompt, model, width, height, negative_prompt, n } = args;
        
        if (!prompt || typeof prompt !== "string") {
          throw new Error("Prompt is required and must be a string");
        }

        const result = await makeImageRequest(prompt, {
          model,
          width,
          height,
          negative_prompt,
          n,
        });

        return {
          content: [
            {
              type: "text",
              text: `Generated ${result.data.length} image(s) using model: ${model || "sd15-onnx"}\nPrompt: "${prompt}"${negative_prompt ? `\nNegative prompt: "${negative_prompt}"` : ""}\nDimensions: ${width || 512}x${height || 512}`,
            },
            ...result.data.map((img, index) => ({
              type: "image",
              data: img.b64_json,
              mimeType: "image/png",
            })),
          ],
        };
      }

      case "recommend_model": {
        const { prompt, priority, hardware } = args;
        
        if (!prompt || typeof prompt !== "string") {
          throw new Error("Prompt is required for model recommendations");
        }

        // Map user-friendly hardware descriptions to our recommendation logic
        const hardwareMap = {
          "cpu_only": { hasGPU: false, memoryConstrained: true },
          "gpu_available": { hasGPU: true, memoryConstrained: false },
          "powerful_gpu": { hasGPU: true, memoryConstrained: false }
        };

        const priorityMap = {
          "quality": { requiresHighQuality: true },
          "speed": { requiresSpeed: true },
          "memory": { memoryConstrained: true }
        };

        const options = {
          ...hardwareMap[hardware] || hardwareMap["cpu_only"],
          ...priorityMap[priority] || {}
        };

        const recommendations = recommendModel(prompt, options);
        
        let responseText = `**Model Recommendations for:** "${prompt}"\n\n`;
        responseText += `**Your Requirements:**\n`;
        responseText += `- Priority: ${priority || "balanced"}\n`;
        responseText += `- Hardware: ${hardware || "cpu_only"}\n\n`;
        
        responseText += `**Recommended Models:**\n\n`;
        recommendations.forEach((rec, index) => {
          responseText += `${index + 1}. **${rec.model}**\n`;
          responseText += `   Reason: ${rec.reason}\n\n`;
        });

        responseText += `**Quick Guide:**\n`;
        responseText += `- **sd15-onnx**: Works on any computer, uses ~500MB RAM, good quality\n`;
        responseText += `- **sdxl-turbo**: Fast + good quality, needs GPU with ~8GB VRAM\n`;
        responseText += `- **sdxl**: Best quality, slower, needs GPU with ~8GB VRAM\n`;
        responseText += `- **flux1-schnell**: Excellent for artistic/complex prompts, needs powerful GPU\n`;

        return {
          content: [
            {
              type: "text",
              text: responseText,
            },
          ],
        };
      }

      case "list_models": {
        const models = await getAvailableModels();
        
        let modelList = "Available Image Generation Models:\n\n";
        
        for (const [modelId, metadata] of Object.entries(models)) {
          modelList += `**${modelId}**\n`;
          if (metadata.display_name) {
            modelList += `  Display Name: ${metadata.display_name}\n`;
          }
          if (metadata.description) {
            modelList += `  Description: ${metadata.description}\n`;
          }
          if (metadata.memory_requirement) {
            modelList += `  Memory: ${metadata.memory_requirement}\n`;
          }
          if (metadata.quantization) {
            modelList += `  Quantization: ${metadata.quantization}\n`;
          }
          modelList += `  Engine: ${metadata.engine}\n`;
          modelList += `  Resolution: ${metadata.min_resolution}-${metadata.max_resolution}px (default: ${metadata.default_resolution}px)\n`;
          modelList += `  Negative Prompts: ${metadata.supports_negative_prompt ? 'Yes' : 'No'}\n\n`;
        }

        return {
          content: [
            {
              type: "text",
              text: modelList,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("ImageAI MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});