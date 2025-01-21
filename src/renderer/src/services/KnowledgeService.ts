import type { ExtractChunkData } from '@llm-tools/embedjs-interfaces'
import AiProvider from '@renderer/providers/AiProvider'
import { FileType, KnowledgeBase, KnowledgeBaseParams, Message } from '@renderer/types'
import { take } from 'lodash'

import { getProviderByModel } from './AssistantService'
import FileManager from './FileManager'

export const getKnowledgeBaseParams = (base: KnowledgeBase): KnowledgeBaseParams => {
  const provider = getProviderByModel(base.model)
  const aiProvider = new AiProvider(provider)

  let host = aiProvider.getBaseURL()

  if (provider.type === 'gemini') {
    host = host + '/v1beta/openai/'
  }

  return {
    id: base.id,
    model: base.model.id,
    dimensions: base.dimensions,
    apiKey: aiProvider.getApiKey() || 'secret',
    apiVersion: provider.apiVersion,
    baseURL: host,
    chunkSize: base.chunkSize || 500,
    chunkOverlap: base.chunkOverlap || 50,
    rerankModel: base.rerankModel?.id || 'BAAI/bge-reranker-v2-m3'
  }
}

export const getFileFromUrl = async (url: string): Promise<FileType | null> => {
  let fileName = ''

  if (url && url.includes('CherryStudio')) {
    if (url.includes('/Data/Files')) {
      fileName = url.split('/Data/Files/')[1]
    }

    if (url.includes('\\Data\\Files')) {
      fileName = url.split('\\Data\\Files\\')[1]
    }
  }

  if (fileName) {
    const fileId = fileName.split('.')[0]
    const file = await FileManager.getFile(fileId)
    if (file) {
      return file
    }
  }

  return null
}

export const getKnowledgeSourceUrl = async (item: ExtractChunkData & { file: FileType | null }) => {
  if (item.metadata.source.startsWith('http')) {
    return item.metadata.source
  }

  if (item.file) {
    return `[${item.file.origin_name}](http://file/${item.file.name})`
  }

  return item.metadata.source
}

export const getKnowledgeReferences = async (base: KnowledgeBase, message: Message) => {
  const searchResults = await window.api.knowledgeBase.search({
    search: message.content,
    base: getKnowledgeBaseParams(base)
  })

  const _searchResults = await Promise.all(
    searchResults.map(async (item) => {
      const file = await getFileFromUrl(item.metadata.source)
      return { ...item, file }
    })
  )

  const references = await Promise.all(
    take(_searchResults, 6).map(async (item, index) => {
      const baseItem = base.items.find((i) => i.uniqueId === item.metadata.uniqueLoaderId)
      return {
        id: index,
        content: item.pageContent,
        sourceUrl: await getKnowledgeSourceUrl(item),
        type: baseItem?.type
      }
    })
  )

  const referencesContent = `\`\`\`json\n${JSON.stringify(references, null, 2)}\n\`\`\``

  return referencesContent
}

export const getRerankResult = async (base: KnowledgeBase, search: string, searchResults: ExtractChunkData[]) => {
  if (!searchResults.length) {
    return searchResults
  }

  try {
    console.log(base)
    const documents = searchResults.map((result) => result.pageContent)
    const rerankModel = base.rerankModel
    const aiProvider = getProviderByModel(rerankModel)
    const response = await fetch(`${aiProvider.apiHost}/v1/rerank`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${aiProvider.apiKey}`
      },
      body: JSON.stringify({
        model: base.rerankModel?.id,
        query: search,
        documents,
        top_n: documents.length,
        return_documents: false,
        max_chunks_per_doc: base.chunkSize,
        overlap_tokens: base.chunkOverlap
      })
    })

    if (!response.ok) {
      console.error('Rerank API error:', await response.text())
      return searchResults
    }

    const rerankResult = await response.json()

    return rerankResult.results.map((result) => {
      const originalResult = searchResults[result.index]
      return {
        ...originalResult,
        score: result.relevance_score
      }
    })
  } catch (error) {
    console.error('Error during reranking:', error)
    return searchResults
  }
}
