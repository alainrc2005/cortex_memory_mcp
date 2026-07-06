import { defineConfig } from 'vitest/config'
import { config as loadEnv } from 'dotenv'

// Carga el .env del proyecto para que los tests de integración
// usen la misma Qdrant URL y API key que el servidor real.
loadEnv()

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 60_000,
    hookTimeout: 30_000,
    include: ['tests/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['src/services/**/*.ts'],
      exclude: ['src/server.ts'],
      thresholds: {
        lines:     60,
        functions: 60,
        branches:  50,
      },
    },
    // Para tests de integración con Qdrant real:
    // Las variables se cargan de .env via loadEnv() arriba.
    // Solo sobreescribimos las que deben ser distintas en tests.
    env: {
      CORTEX_LLM_BACKEND:      process.env.CORTEX_LLM_BACKEND ?? 'none',
      CORTEX_RERANKER_ENABLED: 'false',   // nunca reranker en tests (velocidad)
    },
    // Un solo fork para que no haya condiciones de carrera en Qdrant
    pool: 'forks',
  },
})

