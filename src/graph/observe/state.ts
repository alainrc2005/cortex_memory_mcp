import type { EngramaType } from '../../types/engrama.js'

export interface ObserveState {
  // Input
  content: string
  projectName: string

  // Scoring
  importance: number
  type: EngramaType
  tags: string[]

  // Embedding & Linking
  embedding: number[]
  linkedTo: string[]

  // Output
  engramaId: string
  status: string
  supersededIds: string[]   // IDs marcados como superseded por este engrama
}

export const observeStateChannels = {
  content:     { reducer: (_x: string, y: string) => y, default: () => '' },
  projectName: { reducer: (_x: string, y: string) => y, default: () => 'default' },
  importance:  { reducer: (_x: number, y: number) => y, default: () => 5 },
  type:        { reducer: (_x: EngramaType, y: EngramaType) => y, default: (): EngramaType => 'FACT' },
  tags:        { reducer: (_x: string[], y: string[]) => y, default: (): string[] => [] },
  embedding:   { reducer: (_x: number[], y: number[]) => y, default: (): number[] => [] },
  linkedTo:    { reducer: (_x: string[], y: string[]) => y, default: (): string[] => [] },
  engramaId:      { reducer: (_x: string, y: string) => y, default: () => '' },
  status:         { reducer: (_x: string, y: string) => y, default: () => 'idle' },
  supersededIds:  { reducer: (_x: string[], y: string[]) => y, default: (): string[] => [] },
}
