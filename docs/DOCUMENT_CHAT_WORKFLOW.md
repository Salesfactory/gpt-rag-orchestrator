# Document & Data Upload Workflow

## Overview
This document explains how document uploads (PDFs) and data uploads (spreadsheets) are handled in the unified orchestrator. The system supports two distinct workflows:
- **PDF Documents**: Routed to `document_chat` MCP tool with file ID caching
- **Spreadsheets** (.xlsx, .xls, .csv): Routed to `data_analyst` MCP tool with container thread persistence

Users can attach up to three files in the UI. The backend automatically detects file type, routes to the appropriate tool, reuses cached references when possible, and persists state to Cosmos DB for continuity across conversation turns.

## End-to-End Flow

1. **UI Request**: Sends `question`, `conversation_id`, and `blob_names` (array of document blob names)
2. **Function App**: Extracts `blob_names` and passes to orchestrator
3. **Orchestrator**: Initializes conversation state with `blob_names`
4. **Tool Preparation**: Filters available tools to only `document_chat` when documents present
5. **Tool Planning**: Claude Haiku forced to select `document_chat` tool
6. **Context Injection**: MCP client validates cache and injects context
7. **Tool Execution**: Calls document_chat with documents and optional cached file IDs
8. **Context Extraction**: Parses answer and extracts new file references
9. **Response Generation**: Claude Sonnet generates final response with document context
10. **Persistence**: Saves `uploaded_file_refs` to Cosmos DB for next turn

## Storage Layout

Documents are stored in Azure Blob Storage with this structure:
```
container_name/
  organization_id/
    user_id/
      conversation_id/
        doc1.pdf
        doc2.pdf
        doc3.pdf   # UI enforces max 3 documents
```

## Data Models

### UserUploadedBlobs
Located in `orc/unified_orchestrator/models.py`:
```python
@dataclass
class UserUploadedBlobs:
    kind: str = ""  # "pdf", "spreadsheet", or "unknown"
    items: List[Dict[str, Optional[str]]] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [item.get("blob_name", "") for item in self.items if item.get("blob_name")]
```

### ConversationState
Located in `orc/unified_orchestrator/models.py`:
```python
@dataclass
class ConversationState:
    question: str
    user_uploaded_blobs: UserUploadedBlobs = field(default_factory=UserUploadedBlobs)

    # PDF document caching (document_chat tool)
    uploaded_file_refs: List[Dict[str, str]] = field(default_factory=list)

    # Spreadsheet caching (data_analyst tool)
    cached_dochat_analyst_blobs: List[Dict[str, Optional[str]]] = field(default_factory=list)
    code_thread_id: Optional[str] = None

    # ... other fields
```

### Blob Item Format (Input)
```python
# String format (legacy)
blob_names = ["document.pdf", "data.xlsx"]

# Dict format (with optional file_id for reuse)
blob_names = [
    {"blob_name": "document.pdf", "file_id": "file-abc123"},  # Optional file_id
    {"blob_name": "data.xlsx"}  # file_id omitted
]
```

### File Reference Format (Cached in Cosmos)
```python
# For PDF documents (uploaded_file_refs)
{
    "file_id": "file-abc123",      # OpenAI file ID (expires in 1 hour)
    "blob_name": "document.pdf"    # Original blob name
}

# For spreadsheets (cached_dochat_analyst_blobs)
{
    "blob_name": "data.xlsx",      # Original blob name
    "file_id": "file-xyz789"       # Optional file ID from data_analyst
}
```

## MCP Contract

### Request to document_chat (PDFs)
```python
{
    "question": "Summarize the contract differences",
    "document_names": ["doc1.pdf", "doc2.pdf"],  # Always present
    "cached_file_info": [                         # Only if cache valid
        {"file_id": "file-abc123", "blob_name": "doc1.pdf"},
        {"file_id": "file-def456", "blob_name": "doc2.pdf"}
    ]
}
```

### Response from document_chat
```python
{
    "answer": "Based on the documents...",
    "files": [
        {"file_id": "file-abc123", "blob_name": "doc1.pdf"},
        {"file_id": "file-def456", "blob_name": "doc2.pdf"}
    ],
    "metadata": {...}
}
```

### Request to data_analyst (Spreadsheets)
```python
{
    "query": "Analyze sales trends in this data",
    "organization_id": "org-456",
    "code_thread_id": "container-xyz",  # Reused if blobs haven't changed
    "user_id": "user-123",
    "blob_names": [                      # Optional blob items with file_id
        {"blob_name": "sales.xlsx", "file_id": "file-abc123"}
    ]
}
```

### Response from data_analyst
```python
{
    "success": true,
    "code_thread_id": "container-xyz",  # Persisted for next turn
    "last_agent_message": "Analysis results...",
    "images_processed": [...],           # Chart/visualization metadata
    "blob_urls": [...],                  # Generated artifact URLs
    "error": null
}
```

## Implementation Map

### Entry Point
**File**: `function_app.py`
- Endpoint: `/api/orc` (POST)
- Extracts `blob_names` from request body
- Passes to `ConversationOrchestrator.generate_response_with_progress()`

### Orchestrator
**File**: `orc/unified_orchestrator/orchestrator.py`

**Key Methods**:
- `generate_response_with_progress()`: Main entry point, initializes state with `blob_names`
- `_initialize_node()`: Loads conversation history and cached `uploaded_file_refs`
- `_prepare_tools_node()`: Filters tools to only `document_chat` when `blob_names` present
- `_plan_tools_node()`: Forces Claude to select `document_chat` tool
- `_execute_tools_node()`: Executes the tool via ToolNode
- `_extract_context_node()`: Parses tool results and updates `uploaded_file_refs`

**Blob Normalization** (in `_normalize_blob_inputs`):
```python
def _normalize_blob_inputs(blob_names: Any) -> UserUploadedBlobs:
    """Normalize blob input into structured payload with kind detection."""
    items = []
    for entry in blob_names:
        if isinstance(entry, str):
            items.append({"blob_name": entry, "file_id": None})
        elif isinstance(entry, dict):
            items.append({"blob_name": entry.get("blob_name"), "file_id": entry.get("file_id")})

    kind = _infer_blob_kind([item["blob_name"] for item in items])
    return UserUploadedBlobs(kind=kind, items=items)
```

**Blob Kind Detection** (in `_infer_blob_kind`):
```python
def _infer_blob_kind(blob_names: List[str]) -> str:
    """Determine whether blobs are PDFs or spreadsheets."""
    has_pdf = any(name.lower().endswith(".pdf") for name in blob_names)
    has_sheet = any(name.lower().endswith((".xlsx", ".xls", ".csv")) for name in blob_names)

    if has_sheet:
        return "spreadsheet"  # Spreadsheet takes priority over PDF
    if has_pdf:
        return "pdf"
    return "unknown"
```

**Tool Filtering Logic** (in `_prepare_tools_node`):
```python
is_spreadsheet = state.user_uploaded_blobs.kind == "spreadsheet"

# Exclude document_chat when no PDFs or when spreadsheets present
exclude_doc_chat = len(state.user_uploaded_blobs.names) == 0 or is_spreadsheet

self.wrapped_tools = await self.mcp_client.get_wrapped_tools(
    state=state,
    conversation_history=conversation_history,
    exclude_document_chat=exclude_doc_chat,
)

# Force document_chat for PDFs
if state.user_uploaded_blobs.names and not is_spreadsheet:
    self.wrapped_tools = [t for t in self.wrapped_tools if t.name == "document_chat"]

# Force data_analyst for spreadsheets
elif is_spreadsheet:
    self.wrapped_tools = [t for t in self.wrapped_tools if t.name == "data_analyst"]

# Force data_analyst if mode flag set
elif state.is_data_analyst_mode:
    self.wrapped_tools = [t for t in self.wrapped_tools if t.name == "data_analyst"]
```

**Forced Tool Selection** (in `_plan_tools_node`):
```python
if len(self.wrapped_tools) == 1 and self.wrapped_tools[0].name == "document_chat":
    model_with_tools = self.tool_calling_llm.bind_tools(
        self.wrapped_tools,
        tool_choice={"type": "tool", "name": "document_chat"}
    )
elif len(self.wrapped_tools) == 1 and self.wrapped_tools[0].name == "data_analyst":
    model_with_tools = self.tool_calling_llm.bind_tools(
        self.wrapped_tools,
        tool_choice={"type": "tool", "name": "data_analyst"}
    )
elif len(self.wrapped_tools) == 1 and self.wrapped_tools[0].name == "agentic_search":
    model_with_tools = self.tool_calling_llm.bind_tools(
        self.wrapped_tools,
        tool_choice={"type": "tool", "name": "agentic_search"}
    )
else:
    model_with_tools = self.tool_calling_llm.bind_tools(
        self.wrapped_tools, tool_choice="any"
    )
```

### MCP Client
**File**: `orc/unified_orchestrator/mcp_client.py`

**Key Methods**:
- `connect()`: Establishes SSE connection to MCP server
- `get_wrapped_tools()`: Returns tools wrapped with context injection
- `_create_contextual_tool()`: Wraps MCP tool with automatic context injection

**Cache Validation Logic**:

For **PDFs** (document_chat):
```python
def _validate_blob_names() -> bool:
    """Check if current blob names match cached file refs - document chat only"""
    current_blobs = set(state.user_uploaded_blobs.names)
    cached_blobs = set(
        ref.get("blob_name", "") for ref in state.uploaded_file_refs
    )
    return current_blobs == cached_blobs

# In document_chat context injection (mcp_client.py):
if _validate_blob_names() and state.uploaded_file_refs:
    kwargs["cached_file_info"] = state.uploaded_file_refs
    logger.info(f"Reusing {len(state.uploaded_file_refs)} cached files")
else:
    logger.info(f"Processing {len(state.user_uploaded_blobs.names)} fresh documents")
```

For **Spreadsheets** (data_analyst):
```python
# In _initialize_node (orchestrator.py):
if state.user_uploaded_blobs.kind == "spreadsheet":
    if not self._blob_items_match(
        state.user_uploaded_blobs.items, cached_dochat_analyst_blobs
    ):
        # Invalidate code_thread_id when spreadsheet blobs change
        if code_thread_id:
            logger.info("Spreadsheet blobs changed; invalidating code_thread_id")
        code_thread_id = None

# Blob item matching compares (blob_name, file_id) tuples
def _blob_items_match(items_a, items_b) -> bool:
    """Compare blob item lists ignoring ordering."""
    def normalize(items):
        return set((item["blob_name"], item.get("file_id") or "") for item in items)
    return normalize(items_a) == normalize(items_b)

# In data_analyst context injection (mcp_client.py):
kwargs.update({
    "organization_id": context["organization_id"],
    "code_thread_id": state.code_thread_id,  # Reused if blobs match
    "user_id": context["user_id"],
})
if state.user_uploaded_blobs.items:
    kwargs["blob_names"] = state.user_uploaded_blobs.items
```

### Context Builder
**File**: `orc/unified_orchestrator/context_builder.py`

**Key Method**: `extract_context_from_messages()`
- Parses tool results from LangChain messages
- Extracts context, blob URLs, and file references based on tool type
- Returns tuple: `(context_docs, blob_urls, uploaded_file_refs)`

```python
# For document_chat (PDFs)
elif tool_name == "document_chat" and isinstance(result, dict):
    answer = result.get("answer", result)
    context_docs.append(answer)

    # Extract file references for caching (OpenAI file IDs)
    files = result.get("files", [])
    if files and isinstance(files, list):
        uploaded_file_refs = files

# For data_analyst (Spreadsheets)
elif tool_name == "data_analyst" and isinstance(result, dict):
    last_message = result.get("last_agent_message", result)
    context_docs.append(last_message)

    # Extract blob URLs for charts/visualizations
    result_blob_urls = result.get("blob_urls", [])
    if isinstance(result_blob_urls, list):
        for blob_item in result_blob_urls:
            if isinstance(blob_item, dict):
                blob_path = blob_item.get("blob_path")
                if blob_path:
                    blob_urls.append(blob_path)
                    # Add blob link to context for LLM citation
                    context_docs.append(f"<link>{blob_path}</link>")
```

### State Manager
**File**: `orc/unified_orchestrator/state_manager.py`

**Key Methods**:
- `load_conversation()`: Loads conversation history from Cosmos DB
  - Extracts `code_thread_id`, `last_mcp_tool_used`, `uploaded_file_refs`, and `cached_dochat_analyst_blobs` from most recent assistant message
- `save_conversation()`: Saves updated conversation with metadata
  - For PDFs: saves `uploaded_file_refs` to assistant message
  - For spreadsheets: saves `cached_dochat_analyst_blobs` and `code_thread_id` to assistant message

```python
# Saving PDF file references
if state.uploaded_file_refs:
    assistant_message["uploaded_file_refs"] = state.uploaded_file_refs

# Saving spreadsheet blob cache and code thread
if state.user_uploaded_blobs.kind == "spreadsheet" and state.user_uploaded_blobs.items:
    assistant_message["cached_dochat_analyst_blobs"] = state.user_uploaded_blobs.items

if state.code_thread_id:
    assistant_message["code_thread_id"] = state.code_thread_id
```

## Cache Validation Rules

### PDF Document Cache (document_chat)

**When Cache is Used:**
- Current blob names **exactly match** cached `uploaded_file_refs[*].blob_name`
- Set comparison (order doesn't matter)
- Case-sensitive comparison
- All blob names must match (no partial reuse)

**When Cache is Skipped:**
- Different blob names
- Empty cache (`uploaded_file_refs` is empty)
- Subset or superset of cached files
- Any mismatch triggers fresh processing

**Cache Expiration:**
- OpenAI file IDs expire after 1 hour
- MCP server detects expired IDs and falls back to fresh processing
- Returns new file IDs in response
- Orchestrator overwrites `uploaded_file_refs` with new IDs

### Spreadsheet Cache (data_analyst)

**When Cache is Used (code_thread_id reused):**
- Current blob items **exactly match** cached `cached_dochat_analyst_blobs`
- Compares both `blob_name` and `file_id` as tuples: `(blob_name, file_id)`
- Set comparison (order doesn't matter)
- All blob items must match (no partial reuse)

**When Cache is Invalidated:**
- Different blob items (blob_name or file_id changed)
- Empty cache (`cached_dochat_analyst_blobs` is empty)
- Invalidation sets `code_thread_id = None`, triggering new container creation
- Data analyst creates new container/thread for analysis

**Cache Persistence:**
- `code_thread_id` persists across turns when blob items match
- Allows data analyst to maintain conversation context with same spreadsheet
- `cached_dochat_analyst_blobs` stored in assistant message in Cosmos DB

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HTTP Request                                                 │
│    POST /api/orc                                                │
│    { question, conversation_id, blob_names: ["doc1.pdf"] }     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Function App (function_app.py)                              │
│    - Extract blob_names from request                           │
│    - Pass to orchestrator                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Initialize Node (orchestrator.py)                           │
│    - Load conversation from Cosmos DB                          │
│    - Extract cached uploaded_file_refs                         │
│    - State: { blob_names, uploaded_file_refs }                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Prepare Tools Node (orchestrator.py)                        │
│    - Connect to MCP Server                                     │
│    - Get available tools                                       │
│    - Filter: only document_chat if blob_names present          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Plan Tools Node (orchestrator.py)                           │
│    - Claude Haiku with bind_tools                              │
│    - Force tool_choice: document_chat                          │
│    - Returns AIMessage with tool_calls                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Context Injection (mcp_client.py)                           │
│    - Validate cache: blob_names vs uploaded_file_refs          │
│    - If match: include cached_file_info                        │
│    - If mismatch: omit cache, process fresh                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Execute Tools Node (orchestrator.py)                        │
│    - ToolNode executes document_chat                           │
│    - MCP server processes documents                            │
│    - Returns: { answer, files }                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Extract Context Node (context_builder.py)                   │
│    - Parse tool result from ToolMessage                        │
│    - Extract answer → add to context_docs                      │
│    - Extract files → update uploaded_file_refs                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Generate Response Node (orchestrator.py)                    │
│    - Claude Sonnet with extended thinking                      │
│    - System prompt + document context                          │
│    - Stream response to user                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. Save to Cosmos DB (state_manager.py)                       │
│     - Update conversation history                              │
│     - Save uploaded_file_refs for next turn                    │
│     - Save last_mcp_tool_used: "document_chat"                 │
└─────────────────────────────────────────────────────────────────┘
```

## Spreadsheet vs PDF Workflow Differences

### PDF Documents (document_chat)
1. **File Type Detection**: Files ending in `.pdf`
2. **Tool Selection**: Forced to `document_chat` tool
3. **Caching Mechanism**: OpenAI file IDs cached in `uploaded_file_refs`
4. **Cache Key**: Set of blob names (strings only)
5. **Cache Duration**: 1 hour (OpenAI file ID expiration)
6. **MCP Request**: Includes `document_names` + optional `cached_file_info`

### Spreadsheets (data_analyst)
1. **File Type Detection**: Files ending in `.xlsx`, `.xls`, or `.csv`
2. **Tool Selection**: Forced to `data_analyst` tool
3. **Caching Mechanism**: Container thread ID cached in `code_thread_id`
4. **Cache Key**: Set of (blob_name, file_id) tuples
5. **Cache Duration**: Indefinite (until blob items change)
6. **MCP Request**: Includes `blob_names` (items) + `code_thread_id` for reuse
7. **Special Handling**: Generates charts/visualizations with blob URLs

### Mixed File Types
- If both PDFs and spreadsheets are uploaded, **spreadsheet takes priority**
- System logs warning: "Mixed file types detected; defaulting to spreadsheet handling"
- All files routed to `data_analyst` tool

## Testing Scenarios

### PDF Document Scenarios

#### Scenario 1: First PDF Upload (No Cache)
**Input**:
- `blob_names = ["doc1.pdf"]` or `[{"blob_name": "doc1.pdf"}]`
- `uploaded_file_refs = []` (empty cache)
- `user_uploaded_blobs.kind = "pdf"`

**Expected Behavior**:
- Tool forced: `document_chat`
- Send only `document_names` to MCP
- Tool processes document and returns new file ID
- State updated: `uploaded_file_refs = [{"file_id": "file-abc", "blob_name": "doc1.pdf"}]`
- Saved to Cosmos DB assistant message

#### Scenario 2: Same PDF (Cache Hit)
**Input**:
- `blob_names = ["doc1.pdf"]`
- `uploaded_file_refs = [{"file_id": "file-abc", "blob_name": "doc1.pdf"}]`
- `user_uploaded_blobs.kind = "pdf"`

**Expected Behavior**:
- Cache validation passes (exact match)
- Send `document_names` + `cached_file_info` to MCP
- Tool reuses file ID (no re-upload)
- State updated with returned file refs
- Saved to Cosmos DB

#### Scenario 3: Different PDF (Cache Miss)
**Input**:
- `blob_names = ["doc2.pdf"]`
- `uploaded_file_refs = [{"file_id": "file-abc", "blob_name": "doc1.pdf"}]`
- `user_uploaded_blobs.kind = "pdf"`

**Expected Behavior**:
- Cache validation fails (mismatch)
- Send only `document_names` to MCP (no cache)
- Tool processes new document
- State updated: `uploaded_file_refs = [{"file_id": "file-xyz", "blob_name": "doc2.pdf"}]`
- Saved to Cosmos DB

#### Scenario 4: Multiple PDFs
**Input**:
- `blob_names = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]`
- `uploaded_file_refs = []`
- `user_uploaded_blobs.kind = "pdf"`

**Expected Behavior**:
- All three documents processed
- Returns three file references
- State updated with all three file IDs
- Saved to Cosmos DB

#### Scenario 5: Expired File IDs
**Input**:
- `blob_names = ["doc1.pdf"]`
- `uploaded_file_refs = [{"file_id": "file-expired", "blob_name": "doc1.pdf"}]`
- File ID expired on OpenAI side (>1 hour old)

**Expected Behavior**:
- Cache validation passes (blob names match)
- Send `cached_file_info` to MCP
- MCP server detects expired ID
- Falls back to fresh processing
- Returns new file ID
- State updated with new file ID
- Saved to Cosmos DB

### Spreadsheet Scenarios

#### Scenario 6: First Spreadsheet Upload (No Cache)
**Input**:
- `blob_names = [{"blob_name": "sales.xlsx"}]`
- `cached_dochat_analyst_blobs = []` (empty cache)
- `code_thread_id = None`
- `user_uploaded_blobs.kind = "spreadsheet"`

**Expected Behavior**:
- Tool forced: `data_analyst`
- Send `blob_names` (items) + `code_thread_id=None` to MCP
- Data analyst creates new container and processes data
- Returns new `code_thread_id = "container-abc"`
- State updated:
  - `code_thread_id = "container-abc"`
  - `cached_dochat_analyst_blobs = [{"blob_name": "sales.xlsx", "file_id": "..."}]`
- Saved to Cosmos DB assistant message

#### Scenario 7: Same Spreadsheet (Cache Hit - Thread Reused)
**Input**:
- `blob_names = [{"blob_name": "sales.xlsx", "file_id": "file-123"}]`
- `cached_dochat_analyst_blobs = [{"blob_name": "sales.xlsx", "file_id": "file-123"}]`
- `code_thread_id = "container-abc"`
- `user_uploaded_blobs.kind = "spreadsheet"`

**Expected Behavior**:
- Blob items match → `code_thread_id` NOT invalidated
- Send `blob_names` + `code_thread_id="container-abc"` to MCP
- Data analyst reuses existing container/thread
- Maintains conversation context from previous analysis
- Returns same `code_thread_id`
- State preserved in Cosmos DB

#### Scenario 8: Different Spreadsheet (Cache Miss - Thread Reset)
**Input**:
- `blob_names = [{"blob_name": "revenue.xlsx"}]`
- `cached_dochat_analyst_blobs = [{"blob_name": "sales.xlsx", "file_id": "file-123"}]`
- `code_thread_id = "container-abc"`
- `user_uploaded_blobs.kind = "spreadsheet"`

**Expected Behavior**:
- Blob items mismatch → `code_thread_id` invalidated to `None`
- Send `blob_names` + `code_thread_id=None` to MCP
- Data analyst creates NEW container
- Returns new `code_thread_id = "container-xyz"`
- State updated with new thread and blob cache
- Saved to Cosmos DB

#### Scenario 9: Mixed File Types (Spreadsheet Priority)
**Input**:
- `blob_names = ["doc1.pdf", "sales.xlsx"]`
- `user_uploaded_blobs.kind = "spreadsheet"` (auto-detected)

**Expected Behavior**:
- System logs: "Mixed file types detected; defaulting to spreadsheet handling"
- Tool forced: `data_analyst`
- Both files treated as spreadsheet data
- PDF file may fail processing in data_analyst (expected behavior)

## Example Payloads

### PDF Document Chat Payloads

#### HTTP Request (PDF)
```json
POST /api/orc
{
  "question": "What are the key differences between these contracts?",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "blob_names": ["contract_a.pdf", "contract_b.pdf"],
  "client_principal_id": "user-123",
  "client_principal_name": "John Doe",
  "client_principal_organization": "org-456"
}
```

#### MCP Request to document_chat (Cache Hit)
```json
{
  "question": "What are the key differences between these contracts?",
  "document_names": ["contract_a.pdf", "contract_b.pdf"],
  "cached_file_info": [
    {"file_id": "file-abc123", "blob_name": "contract_a.pdf"},
    {"file_id": "file-def456", "blob_name": "contract_b.pdf"}
  ]
}
```

#### MCP Request to document_chat (Cache Miss)
```json
{
  "question": "What are the key differences between these contracts?",
  "document_names": ["contract_a.pdf", "contract_b.pdf"]
}
```

#### MCP Response from document_chat
```json
{
  "answer": "Based on the analysis of both contracts, the key differences are:\n1. Payment terms...\n2. Termination clauses...",
  "files": [
    {"file_id": "file-abc123", "blob_name": "contract_a.pdf"},
    {"file_id": "file-def456", "blob_name": "contract_b.pdf"}
  ],
  "metadata": {
    "processing_time": 2.5,
    "pages_analyzed": 45
  }
}
```

### Spreadsheet Data Analysis Payloads

#### HTTP Request (Spreadsheet)
```json
POST /api/orc
{
  "question": "Show me the sales trends for Q4",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "blob_names": [
    {"blob_name": "sales_q4.xlsx", "file_id": "file-xyz789"}
  ],
  "client_principal_id": "user-123",
  "client_principal_name": "John Doe",
  "client_principal_organization": "org-456"
}
```

#### MCP Request to data_analyst (First Upload)
```json
{
  "query": "Show me the sales trends for Q4",
  "organization_id": "org-456",
  "code_thread_id": null,
  "user_id": "user-123",
  "blob_names": [
    {"blob_name": "sales_q4.xlsx", "file_id": null}
  ]
}
```

#### MCP Request to data_analyst (Thread Reuse)
```json
{
  "query": "Show me the sales trends for Q4",
  "organization_id": "org-456",
  "code_thread_id": "container-abc123",
  "user_id": "user-123",
  "blob_names": [
    {"blob_name": "sales_q4.xlsx", "file_id": "file-xyz789"}
  ]
}
```

#### MCP Response from data_analyst
```json
{
  "success": true,
  "code_thread_id": "container-abc123",
  "last_agent_message": "Here are the Q4 sales trends analysis...",
  "images_processed": [
    {
      "path": "artifacts/chart_sales_trend.png",
      "description": "Sales trend chart for Q4"
    }
  ],
  "blob_urls": [
    {
      "blob_path": "https://storage.blob.core.windows.net/artifacts/chart_sales_trend.png",
      "blob_name": "chart_sales_trend.png"
    }
  ],
  "error": null
}
```

## Notes and Limitations

### General
- **File Limit**: UI enforces maximum 3 files; backend accepts any length
- **Case Sensitivity**: Blob name comparison is case-sensitive
- **No Partial Reuse**: Any cache mismatch triggers full reprocessing
- **Local Testing**: `function_app.py` may include default `blob_names` for local dev

### PDF Documents (document_chat)
- **File ID Expiration**: OpenAI file IDs expire after 1 hour
- **Overwrite Strategy**: Always overwrite `uploaded_file_refs` with latest from tool
- **Cache Key**: Set of blob names (strings only)
- **Supported Formats**: `.pdf` files only

### Spreadsheets (data_analyst)
- **Code Thread Persistence**: `code_thread_id` persists indefinitely until blob items change
- **Cache Invalidation**: Any change to blob items (name or file_id) invalidates thread
- **Overwrite Strategy**: Always overwrite `cached_dochat_analyst_blobs` with current items
- **Cache Key**: Set of (blob_name, file_id) tuples
- **Supported Formats**: `.xlsx`, `.xls`, `.csv` files
- **Visualization Support**: Can generate charts and return blob URLs

### Mixed File Types
- **Spreadsheet Priority**: When both PDFs and spreadsheets are uploaded, all files route to `data_analyst`
- **PDF Handling Warning**: PDFs may fail processing when routed to `data_analyst`
- **Recommendation**: Avoid mixing file types in single upload

## Related Files

- `function_app.py` - HTTP endpoint, extracts `blob_names` from request
- `orc/unified_orchestrator/orchestrator.py` - Main orchestrator and workflow nodes
  - `_normalize_blob_inputs()` - Blob normalization to UserUploadedBlobs
  - `_infer_blob_kind()` - File type detection (PDF vs spreadsheet)
  - `_blob_items_match()` - Blob item comparison for spreadsheet cache
  - `_initialize_node()` - Loads cached metadata from Cosmos
  - `_prepare_tools_node()` - Tool filtering based on file type
  - `_plan_tools_node()` - Forced tool selection
  - `_extract_context_node()` - Extracts metadata from tool results
- `orc/unified_orchestrator/models.py` - Data models
  - `UserUploadedBlobs` - Normalized blob metadata with kind field
  - `ConversationState` - State object with upload caching fields
- `orc/unified_orchestrator/mcp_client.py` - MCP connection and context injection
  - `_create_contextual_tool()` - Wraps tools with context injection
  - `_validate_blob_names()` - PDF cache validation
  - Context injection for `document_chat` and `data_analyst`
- `orc/unified_orchestrator/context_builder.py` - Context extraction from tool results
  - `extract_context_from_messages()` - Extracts context, blob URLs, file refs
- `orc/unified_orchestrator/state_manager.py` - Cosmos DB persistence
  - `load_conversation()` - Loads cached metadata
  - `save_conversation()` - Saves file refs and code thread ID
- `orc/unified_orchestrator/utils.py` - Utility functions
  - `transform_artifacts_to_images()` - Process data analyst artifacts
  - `transform_artifacts_to_blobs()` - Extract blob URLs

## Status

✅ **Fully Implemented** - Both PDF document chat and spreadsheet data analysis workflows are live in production as of the unified orchestrator architecture (v2.0+).

### Recent Updates
- **Spreadsheet Support**: Added data analyst integration for .xlsx, .xls, .csv files
- **Blob Kind Detection**: Automatic file type detection with priority handling
- **Dual Caching**: Separate cache mechanisms for PDFs (OpenAI file IDs) and spreadsheets (code thread IDs)
- **Enhanced Blob Format**: Support for both string and dict blob_names with optional file_id
