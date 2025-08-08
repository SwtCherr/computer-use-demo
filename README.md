## Getting Started

### Prerequisites

- Anaconda installed
- Nebius API key (get it from [Nebius AI](https://nebius.com))

### Installation & Running

```bash
# Create and activate сonda environment
conda create -n nebius_env python=3.12 -y
conda activate nebius_env

# Install dependencies
pip install -r computer_use_demo/requirements.txt

# Set API key
export NEBIUS_API_KEY='your_key'

# Launch the application
python -m streamlit run computer_use_demo/streamlit.py
```


## Adaptation Process

### Implementation Changes

#### 1. API Client Transition
The code was adapted to replace the multi-provider Anthropic architecture with Nebius AI's OpenAI-compatible API. This simplification eliminated provider-specific authentication checks and configuration logic.

#### 2. Message Format Conversion
The message structure was adapted to Nebius AI's API requirements. The system prompt is now included directly within the conversation history as a dedicated system message, rather than being passed separately to the API.

#### 3. Anthropic-Specific Feature Removal
Features incompatible with Nebius API were deprecated:
- Advanced caching mechanisms
- Specialized thinking parameters

#### 4. Error Handling Standardization
Exception handling was generalized by removing provider-specific error cases. A consistent error reporting approach was implemented across all potential failure scenarios, simplifying UI error display.

#### 5. Updating the User Interface
The Streamlit interface was modified to remove provider selection controls and disable tool-related configuration options. Nebius-specific parameters were added.


### Main Challenge: Tool Implementation Limitations

The adaptation currently lacks tool implementation support due to two distinct technical challenges:

#### 1. Tool Conversion Difficulties

Initial attempts to convert predefined Anthropic tools to OpenAI-compatible format proved unsuccessful. While this doesn't indicate fundamental incompatibility, the current implementation approach failed to produce functional results. Further investigation is needed to determine whether schema adaptation or complete reimplementation will resolve this issue.

#### 2. Model Capability Constraints

Vision-capable models available through Nebius API fundamentally lack support for automatic tool selection (tool_choice='auto'). Successful tool implementation will require explicit tool specification (tool_choice={'type': 'function', 'function': {'name': 'specific_tool'}}), significantly altering the original tool invocation approach.


## Evaluation Metrics

To comprehensively assess the agent's operational effectiveness, I would prioritize the following key performance metrics:

### 1. Task Completion Rate
Measures the percentage of successfully completed user requests. 

### 2. Tool Utilization Accuracy
Measures correct tool selection and proper argument usage. Calculated as the percentage of appropriate tool choices relative to total tool invocations. 

### 3. Conversation Efficiency
Tracks the average number of user-agent exchanges needed to complete tasks.

### 4. Latency Performance
Measures end-to-end response time from query submission to final result delivery. 

### 5. Safety Compliance
Calculates the percentage of operations without security violations.
