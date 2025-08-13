# SL2 AI Report Generator API Documentation

## Overview

The SL2 AI Report Generator API is an intelligent code generation system that converts natural language queries into executable JavaScript code for the SL2 platform. It uses advanced Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) to produce both server-side data retrieval scripts and client-side visualization code.

## Main Goals

### Primary Objectives
1. **Natural Language to Code Translation**: Convert human-readable queries into functional JavaScript code
2. **Automated Report Generation**: Generate complete reports with both data retrieval and visualization components
3. **Context-Aware Code Generation**: Leverage metadata, API documentation, and examples to produce accurate code
4. **Multi-Model Support**: Support various LLM backends (local models, OpenAI, Claude) for flexible deployment
5. **Interactive Development**: Provide real-time code generation for rapid prototyping and development

### Key Features
- **Hybrid Search**: Combines multiple knowledge sources for comprehensive context retrieval
- **Dual Script Generation**: Produces both server-side (data) and client-side (visualization) JavaScript code
- **Metadata Integration**: Uses SL2 system metadata to ensure code accuracy and compatibility
- **Example-Driven Learning**: Leverages existing report examples to improve code quality
- **Validation & Testing**: Includes code validation and error handling mechanisms
- **API-First Design**: RESTful API for easy integration with other systems

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Browser   │    │  External APIs  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │              HTTP/REST API Calls            │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Flask API Server      │
                    │   (api_server.py)         │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   JS Code Generator       │
                    │ (js_code_generator.py)    │
                    └─────────────┬─────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼───────┐
   │   RAG Manager   │  │  LLM Processor  │  │   Validators  │
   │ (rag_manager.py)│  │    (Multiple)   │  │   & Utils     │
   └────────┬────────┘  └─────────────────┘  └───────────────┘
            │
    ┌───────┼───────┐
    │       │       │
┌───▼───┐ ┌─▼─┐ ┌───▼───┐
│Metadata│ │API│ │Examples│
│Indexer │ │Idx│ │Indexers│
└───────┘ └───┘ └───────┘
```

### Core Components

#### 1. API Server (`api_server.py`)
- **Purpose**: RESTful API interface for all system interactions
- **Key Endpoints**:
  - `/generate` - Full code generation (server + client)
  - `/generate-prompt` - Prompt generation only
  - `/generate-client-script-prompt` - Client-side prompt generation
  - `/chat` - Direct LLM interaction
  - `/update-metadata` - Dynamic metadata updates
  - `/health` - System health monitoring

#### 2. JavaScript Code Generator (`js_code_generator.py`)
- **Purpose**: Main orchestrator for code generation process
- **Responsibilities**:
  - Query processing and enhancement
  - Context retrieval coordination
  - Prompt construction for server and client scripts
  - Code extraction and validation
  - Result formatting and response preparation

#### 3. RAG Manager (`rag_manager.py`)
- **Purpose**: Retrieval-Augmented Generation coordinator
- **Knowledge Sources**:
  - **Metadata Indexer**: SL2 system models and fields
  - **JS API Indexer**: Available JavaScript API methods
  - **Report Example Indexers**: Historical report patterns
  - **Chart Example Indexers**: Visualization templates
- **Search Strategy**: Hybrid search combining multiple sources with reranking

#### 4. LLM Processors
- **Local Processor** (`llm_processor.py`): Local model execution with GPU optimization
- **OpenAI Processor** (`llm_openai_processor.py`): OpenAI API integration
- **Claude Processor** (`llm_claude_processor.py`): Anthropic Claude integration

#### 5. Indexing System
Multiple specialized indexers for different knowledge types:
- **MetadataIndexer**: SL2 data models and field definitions
- **JsApiIndexer**: JavaScript API documentation
- **ReportExampleIndexer**: Complete report examples
- **ReportDataExampleIndexer**: Server-side data patterns
- **ReportChartExampleIndexer**: Client-side visualization patterns

## Data Flow

### Code Generation Flow
```
1. Natural Language Query
   ↓
2. Query Enhancement & Processing
   ↓
3. Context Retrieval (RAG)
   ├── Metadata Search
   ├── API Documentation Search
   ├── Example Pattern Search
   └── Chart Template Search
   ↓
4. Prompt Construction
   ├── Server Script Prompt
   └── Client Script Prompt
   ↓
5. LLM Processing
   ├── Server Code Generation
   └── Client Code Generation
   ↓
6. Code Extraction & Validation
   ↓
7. Response Formatting
   ↓
8. API Response
```

### Knowledge Sources Integration
```
SL2 Metadata ─┐
              ├─► Hybrid Search ─► Context Selection ─► Prompt Generation
JS API Docs ──┤
              │
Examples ─────┘
```

## API Endpoints

### Core Generation Endpoints

#### POST `/generate`
**Purpose**: Generate complete JavaScript solution (server + client scripts)
**Input**: Natural language query
**Output**: Server script, client script, validation results, context information

#### POST `/generate-prompt`
**Purpose**: Generate contextual prompt for server-side development
**Input**: Natural language query
**Output**: Formatted prompt with relevant context

#### POST `/generate-client-script-prompt`
**Purpose**: Generate prompt for client-side visualization development
**Input**: Query + optional server script context
**Output**: Client-focused prompt with chart examples

### Utility Endpoints

#### POST `/chat`
**Purpose**: Direct LLM interaction with conversation history
**Input**: Prompt + optional conversation history
**Output**: LLM response with parameters

#### POST `/update-metadata`
**Purpose**: Dynamic metadata management
**Input**: New metadata + action (add/replace)
**Output**: Update status and statistics

#### GET `/health`
**Purpose**: System health and configuration monitoring
**Output**: System status and model information

## Configuration & Deployment

### Environment Variables
- **LLM_PROCESSOR_TYPE**: Choose processor type (`local`, `openai`, `claude`)
- **MODEL_KEY**: Local model selection
- **OPENAI_API_KEY**: OpenAI authentication
- **CLAUDE_API_KEY**: Claude authentication
- **METADATA_PATH**: System metadata file location
- **PORT**: API server port

### Supported Models
- **Local**: CodeLlama, Mixtral, Qwen, DeepSeek variants
- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo variants
- **Claude**: Claude-3 Haiku, Sonnet, Opus, Claude-3.5 Sonnet

## Use Cases

### Primary Use Cases
1. **Interactive Report Building**: Users describe desired reports in natural language
2. **Data Exploration**: Quick code generation for data analysis tasks
3. **Visualization Creation**: Automated chart and graph generation
4. **API Learning**: Contextual examples for SL2 JavaScript API usage
5. **Rapid Prototyping**: Fast iteration on report and dashboard concepts

### Integration Scenarios
- **Development IDEs**: Code completion and suggestion systems
- **Business Intelligence Tools**: Natural language query interfaces
- **Training Systems**: Learning platforms for SL2 development
- **Automated Reporting**: Scheduled report generation systems

## Technical Specifications

### Performance Characteristics
- **Response Time**: 2-10 seconds depending on model and complexity
- **Concurrency**: Supports multiple simultaneous requests
- **Memory Requirements**: 8-16GB RAM for local models
- **GPU Support**: CUDA optimization for local model acceleration

### Quality Assurance
- **Code Validation**: Syntax and pattern validation
- **Context Relevance**: RAG-based context scoring
- **Example Alignment**: Pattern matching with known solutions
- **Error Handling**: Comprehensive error reporting and recovery

## Future Enhancements

### Planned Features
1. **Interactive Debugging**: Step-by-step code explanation and debugging
2. **Code Optimization**: Performance and efficiency improvements
3. **Multi-Language Support**: Extension to other programming languages
4. **Advanced Analytics**: Usage patterns and success metrics
5. **Custom Model Training**: Fine-tuning on SL2-specific patterns

### Scalability Roadmap
- **Distributed Processing**: Multi-instance deployment support
- **Caching Layer**: Response caching for common queries
- **Load Balancing**: Request distribution across multiple processors
- **Monitoring & Observability**: Comprehensive system monitoring
