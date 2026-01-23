/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.opensearch.ml.common.conversation.ActionConstants.ADDITIONAL_INFO_FIELD;
import static org.opensearch.ml.common.conversation.ActionConstants.AI_RESPONSE_FIELD;
import static org.opensearch.ml.common.utils.StringUtils.gson;
import static org.opensearch.ml.common.utils.StringUtils.processTextDoc;
import static org.opensearch.ml.common.utils.ToolUtils.filterToolOutput;
import static org.opensearch.ml.common.utils.ToolUtils.getToolName;
import static org.opensearch.ml.common.utils.ToolUtils.parseResponse;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.DISABLE_TRACE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.INTERACTIONS_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_CHAT_HISTORY_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_SUFFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.RESPONSE_FORMAT_INSTRUCTION;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALL_ID;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_RESPONSE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_RESULT;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.VERBOSE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.cleanUpResource;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.constructToolParams;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.createMemoryParams;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.createTools;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.extractContentBlocksFromParameters;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getCurrentDateTime;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMcpToolSpecs;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMessageHistoryLimit;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMlToolSpecs;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getToolNames;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.outputToOutputString;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.parseLLMOutput;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.substitute;
import static org.opensearch.ml.engine.algorithms.agent.PromptTemplate.CHAT_HISTORY_PREFIX;
import static org.opensearch.ml.engine.tools.ReadFromScratchPadTool.SCRATCHPAD_NOTES_KEY;

import java.lang.reflect.Type;
import java.security.PrivilegedActionException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import com.google.gson.reflect.TypeToken;
import org.apache.commons.text.StringEscapeUtils;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.StepListener;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.MLMemoryType;
import org.opensearch.ml.common.agent.ContentBlock;
import org.opensearch.ml.common.agent.LLMSpec;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.conversation.Interaction;
import org.opensearch.ml.common.memory.Memory;
import org.opensearch.ml.common.memory.Message;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.encryptor.Encryptor;
import org.opensearch.ml.engine.function_calling.FunctionCalling;
import org.opensearch.ml.engine.function_calling.FunctionCallingFactory;
import org.opensearch.ml.engine.function_calling.LLMMessage;
import org.opensearch.ml.engine.memory.ConversationIndexMessage;
import org.opensearch.ml.engine.tools.MLModelTool;
import org.opensearch.remote.metadata.client.SdkClient;
import org.opensearch.transport.TransportChannel;
import org.opensearch.transport.client.Client;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Data
@NoArgsConstructor
public class MLChatAgentRunner implements MLAgentRunner {

    public static final String SESSION_ID = "session_id";
    public static final String LLM_TOOL_PROMPT_PREFIX = "LanguageModelTool.prompt_prefix";
    public static final String LLM_TOOL_PROMPT_SUFFIX = "LanguageModelTool.prompt_suffix";
    public static final String TOOLS = "tools";
    public static final String TOOL_DESCRIPTIONS = "tool_descriptions";
    public static final String TOOL_NAMES = "tool_names";
    public static final String OS_INDICES = "opensearch_indices";
    public static final String EXAMPLES = "examples";
    public static final String SCRATCHPAD = "scratchpad";
    public static final String CHAT_HISTORY = "chat_history";
    public static final String NEW_CHAT_HISTORY = "_chat_history";
    public static final String CONTEXT = "context";
    public static final String PROMPT = "prompt";
    public static final String LLM_RESPONSE = "llm_response";
    public static final String MAX_ITERATION = "max_iteration";
    public static final String THOUGHT = "thought";
    public static final String ACTION = "action";
    public static final String ACTION_INPUT = "action_input";
    public static final String FINAL_ANSWER = "final_answer";
    public static final String THOUGHT_RESPONSE = "thought_response";
    public static final String INTERACTIONS = "_interactions";
    public static final String INTERACTION_TEMPLATE_TOOL_RESPONSE = "interaction_template.tool_response";
    public static final String CHAT_HISTORY_QUESTION_TEMPLATE = "chat_history_template.user_question";
    public static final String CHAT_HISTORY_RESPONSE_TEMPLATE = "chat_history_template.ai_response";
    public static final String CHAT_HISTORY_MULTIMODAL_QUESTION_TEMPLATE = "chat_history_multimodal_question_template";
    public static final String CHAT_HISTORY_MESSAGE_PREFIX = "${_chat_history.message.";
    public static final String LLM_INTERFACE = "_llm_interface";
    public static final String INJECT_DATETIME_FIELD = "inject_datetime";
    public static final String DATETIME_FORMAT_FIELD = "datetime_format";
    public static final String SYSTEM_PROMPT_FIELD = "system_prompt";
    private static final String DEFAULT_SYSTEM_PROMPT = "You are an helpful assistant."; // empty system prompt

    private static final String DEFAULT_MAX_ITERATIONS = "10";
    private static final String MAX_ITERATIONS_MESSAGE = "Agent reached maximum iterations (%d) without completing the task";

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory.Factory> memoryFactoryMap;
    private SdkClient sdkClient;
    private Encryptor encryptor;
    private StreamingWrapper streamingWrapper;

    public MLChatAgentRunner(
        Client client,
        Settings settings,
        ClusterService clusterService,
        NamedXContentRegistry xContentRegistry,
        Map<String, Tool.Factory> toolFactories,
        Map<String, Memory.Factory> memoryFactoryMap,
        SdkClient sdkClient,
        Encryptor encryptor
    ) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryFactoryMap = memoryFactoryMap;
        this.sdkClient = sdkClient;
        this.encryptor = encryptor;
    }

    @Override
    public void run(MLAgent mlAgent, Map<String, String> inputParams, ActionListener<Object> listener, TransportChannel channel) {
        this.streamingWrapper = new StreamingWrapper(channel, client);
        Map<String, String> params = new HashMap<>();
        if (mlAgent.getParameters() != null) {
            params.putAll(mlAgent.getParameters());
            for (String key : mlAgent.getParameters().keySet()) {
                if (key.startsWith("_")) {
                    params.put(key, mlAgent.getParameters().get(key));
                }
            }
        }

        params.putAll(inputParams);

        String llmInterface = params.get(LLM_INTERFACE);
        FunctionCalling functionCalling = FunctionCallingFactory.create(llmInterface);
        if (functionCalling != null) {
            functionCalling.configure(params);
        }

        String memoryType = MLMemoryType.from(mlAgent.getMemory().getType()).name();
        String memoryId = params.get(MLAgentExecutor.MEMORY_ID);
        String appType = mlAgent.getAppType();
        String title = params.get(MLAgentExecutor.QUESTION);
        String chatHistoryPrefix = params.getOrDefault(PROMPT_CHAT_HISTORY_PREFIX, CHAT_HISTORY_PREFIX);
        String chatHistoryQuestionTemplate = params.get(CHAT_HISTORY_QUESTION_TEMPLATE);
        String chatHistoryResponseTemplate = params.get(CHAT_HISTORY_RESPONSE_TEMPLATE);
        int messageHistoryLimit = getMessageHistoryLimit(params);

        Memory.Factory<Memory<Interaction, ?, ?>> memoryFactory = memoryFactoryMap.get(memoryType);

        Map<String, Object> memoryParams = createMemoryParams(title, memoryId, appType, mlAgent);
        memoryFactory.create(memoryParams, ActionListener.wrap(memory -> {
            // TODO: call runAgent directly if messageHistoryLimit == 0
            memory.getMessages(messageHistoryLimit, ActionListener.wrap((List<Interaction> r) -> {
                List<Message> messageList = new ArrayList<>();
                for (Interaction next : r) {
                    String response = next.getResponse();
                    // As we store the conversation with empty response first and then update when have final answer,
                    // filter out those in-flight requests when run in parallel
                    if (Strings.isNullOrEmpty(response)) {
                        continue;
                    }
                    // Keep the original Interaction object to preserve additionalInfo with content
                    // blocks
                    messageList.add(next);
                }

                if (!messageList.isEmpty()) {
                    if (chatHistoryQuestionTemplate == null) {
                        StringBuilder chatHistoryBuilder = new StringBuilder();
                        chatHistoryBuilder.append(chatHistoryPrefix);
                        for (Message message : messageList) {
                            chatHistoryBuilder.append(message.toString()).append("\n");
                        }
                        params.put(CHAT_HISTORY, chatHistoryBuilder.toString());

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added to input params to validate
                        inputParams.put(CHAT_HISTORY, chatHistoryBuilder.toString());
                    } else {
                        List<String> chatHistory = new ArrayList<>();
                        for (Message message : messageList) {
                            // Since we now preserve Interaction objects, we can safely cast
                            Interaction interaction = (Interaction) message;
                            Map<String, String> messageParams = new HashMap<>();
                            messageParams.put("question", processTextDoc(interaction.getInput()));
                            // Check if multi-modal content blocks are available in additionalInfo
                            String contentBlocksJson = null;
                            if (interaction.getAdditionalInfo() != null) {
                                contentBlocksJson = interaction.getAdditionalInfo().get("input_content_blocks");
                            }

                            // Determine which question template to use based on content availability
                            boolean hasContentBlocks = (contentBlocksJson != null && !contentBlocksJson.isEmpty());
                            String multimodalQuestionTemplate = params.get(CHAT_HISTORY_MULTIMODAL_QUESTION_TEMPLATE);
                            boolean hasMultimodalQuestionTemplate = (multimodalQuestionTemplate != null);

                            String questionTemplate = chatHistoryQuestionTemplate;
                            if (hasContentBlocks && hasMultimodalQuestionTemplate) {
                                // Use multi-modal question template when content blocks are available and
                                // template exists
                                questionTemplate = multimodalQuestionTemplate;

                                try {
                                    // Parse content blocks from JSON
                                    Type listType = new TypeToken<List<ContentBlock>>() {}.getType();
                                    List<ContentBlock> contentBlocks = gson.fromJson(contentBlocksJson, listType);
                                    // Convert content blocks to Bedrock Converse format
                                    String contentBlocksForTemplate = convertContentBlocksToBedrockFormat(contentBlocks);
                                    // Fallback to text-only if conversion fails
                                    messageParams
                                        .put(
                                            "content_blocks",
                                            Objects
                                                .requireNonNullElseGet(
                                                    contentBlocksForTemplate,
                                                    () -> "[{\"text\":\""
                                                        + StringEscapeUtils.escapeJson(processTextDoc(interaction.getInput()))
                                                        + "\"}]"
                                                )
                                        );
                                } catch (Exception e) {
                                    log.warn("Failed to parse content blocks from chat history, falling back to text", e);
                                    // Fallback to text-only content block
                                    messageParams
                                        .put(
                                            "content_blocks",
                                            "[{\"text\":\"" + StringEscapeUtils.escapeJson(processTextDoc(interaction.getInput())) + "\"}]"
                                        );
                                }
                            } else {
                                // Use regular text-only templates
                                messageParams.put("question", processTextDoc(interaction.getInput()));
                            }

                            StringSubstitutor substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatQuestionMessage = substitutor.replace(questionTemplate);
                            chatHistory.add(chatQuestionMessage);

                            messageParams.clear();
                            messageParams.put("response", processTextDoc(interaction.getResponse()));
                            substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatResponseMessage = substitutor.replace(chatHistoryResponseTemplate);
                            chatHistory.add(chatResponseMessage);
                        }
                        params.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                        params.put(NEW_CHAT_HISTORY, String.join(", ", chatHistory) + ", ");

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added
                        // to input params to validate
                        inputParams.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                    }
                }

                runAgent(mlAgent, params, listener, memory, memory.getId(), functionCalling);
            }, e -> {
                log.error("Failed to get chat history", e);
                listener.onFailure(e);
            }));
        }, listener::onFailure));
    }

    private void runAgent(
        MLAgent mlAgent,
        Map<String, String> params,
        ActionListener<Object> listener,
        Memory memory,
        String sessionId,
        FunctionCalling functionCalling
    ) {
        List<MLToolSpec> toolSpecs = getMlToolSpecs(mlAgent, params);

        // Create a common method to handle both success and failure cases
        Consumer<List<MLToolSpec>> processTools = (allToolSpecs) -> {
            Map<String, Tool> tools = new HashMap<>();
            Map<String, MLToolSpec> toolSpecMap = new HashMap<>();
            createTools(toolFactories, params, allToolSpecs, tools, toolSpecMap, mlAgent);
            runReAct(mlAgent.getLlm(), tools, toolSpecMap, params, memory, sessionId, mlAgent.getTenantId(), listener, functionCalling);
        };

        // Fetch MCP tools and handle both success and failure cases
        getMcpToolSpecs(mlAgent, client, sdkClient, encryptor, ActionListener.wrap(mcpTools -> {
            toolSpecs.addAll(mcpTools);
            processTools.accept(toolSpecs);
        }, e -> {
            log.error("Failed to get MCP tools, continuing with base tools only", e);
            processTools.accept(toolSpecs);
        }));
    }

    private void runReAct(
        LLMSpec llm,
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> parameters,
        Memory memory,
        String sessionId,
        String tenantId,
        ActionListener<Object> listener,
        FunctionCalling functionCalling
    ) {
        Map<String, String> tmpParameters = constructLLMParams(llm, parameters);
        String prompt = constructLLMPrompt(tools, tmpParameters);
        tmpParameters.put(PROMPT, prompt);
        final String finalPrompt = prompt;

        String question = tmpParameters.get(MLAgentExecutor.QUESTION);
        String parentInteractionId = tmpParameters.get(MLAgentExecutor.PARENT_INTERACTION_ID);
        boolean verbose = Boolean.parseBoolean(tmpParameters.getOrDefault(VERBOSE, "false"));
        boolean traceDisabled = tmpParameters.containsKey(DISABLE_TRACE) && Boolean.parseBoolean(tmpParameters.get(DISABLE_TRACE));

        // Extract content blocks for multi-modal memory storage
        final List<ContentBlock> inputContentBlocks = extractContentBlocksFromParameters(tmpParameters);

        // Create root interaction.
        // Trace number
        AtomicInteger traceNumber = new AtomicInteger(0);

        AtomicReference<StepListener<MLTaskResponse>> lastLlmListener = new AtomicReference<>();
        AtomicReference<String> lastThought = new AtomicReference<>();
        AtomicReference<String> lastAction = new AtomicReference<>();
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        AtomicReference<String> lastToolSelectionResponse = new AtomicReference<>();
        Map<String, Object> additionalInfo = new ConcurrentHashMap<>();
        Map<String, String> lastToolParams = new ConcurrentHashMap<>();

        StepListener firstListener = new StepListener<MLTaskResponse>();
        lastLlmListener.set(firstListener);
        StepListener<?> lastStepListener = firstListener;

        StringBuilder scratchpadBuilder = new StringBuilder();
        List<String> interactions = new CopyOnWriteArrayList<>();

        StringSubstitutor tmpSubstitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
        AtomicReference<String> newPrompt = new AtomicReference<>(tmpSubstitutor.replace(prompt));
        tmpParameters.put(PROMPT, newPrompt.get());
        List<ModelTensors> traceTensors = createModelTensors(sessionId, parentInteractionId);
        int maxIterations = Integer.parseInt(tmpParameters.getOrDefault(MAX_ITERATION, DEFAULT_MAX_ITERATIONS));
        for (int i = 0; i < maxIterations; i++) {
            int finalI = i;
            StepListener<?> nextStepListener = (i == maxIterations - 1) ? null : new StepListener<>();

            lastStepListener.whenComplete(output -> {
                StringBuilder sessionMsgAnswerBuilder = new StringBuilder();
                if (finalI % 2 == 0) {
                    MLTaskResponse llmResponse = (MLTaskResponse) output;
                    ModelTensorOutput tmpModelTensorOutput = (ModelTensorOutput) llmResponse.getOutput();
                    List<String> llmResponsePatterns = gson.fromJson(tmpParameters.get("llm_response_pattern"), List.class);
                    Map<String, String> modelOutput = parseLLMOutput(
                        parameters,
                        tmpModelTensorOutput,
                        llmResponsePatterns,
                        tools.keySet(),
                        interactions,
                        functionCalling
                    );

                    streamingWrapper.fixInteractionRole(interactions);
                    String thought = String.valueOf(modelOutput.get(THOUGHT));
                    String toolCallId = String.valueOf(modelOutput.get("tool_call_id"));
                    String action = String.valueOf(modelOutput.get(ACTION));
                    String actionInput = String.valueOf(modelOutput.get(ACTION_INPUT));
                    String thoughtResponse = modelOutput.get(THOUGHT_RESPONSE);
                    String finalAnswer = modelOutput.get(FINAL_ANSWER);

                    if (finalAnswer != null) {
                        finalAnswer = finalAnswer.trim();
                        sendFinalAnswer(
                            sessionId,
                            listener,
                            question,
                            parentInteractionId,
                            verbose,
                            traceDisabled,
                            traceTensors,
                            memory,
                            traceNumber,
                            additionalInfo,
                            finalAnswer,
                            inputContentBlocks
                        );
                        cleanUpResource(tools);
                        return;
                    }

                    sessionMsgAnswerBuilder.append(thought);
                    lastThought.set(thought);
                    lastAction.set(action);
                    lastActionInput.set(actionInput);
                    lastToolSelectionResponse.set(thoughtResponse);

                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(List.of(ModelTensor.builder().name("response").result(thoughtResponse).build()))
                                .build()
                        );

                    saveTraceData(
                        memory,
                        memory.getType(),
                        question,
                        thoughtResponse,
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        "LLM",
                        inputContentBlocks
                    );

                    if (nextStepListener == null) {
                        handleMaxIterationsReached(
                            sessionId,
                            listener,
                            question,
                            parentInteractionId,
                            verbose,
                            traceDisabled,
                            traceTensors,
                            memory,
                            traceNumber,
                            additionalInfo,
                            lastThought,
                            maxIterations,
                            tools,
                            inputContentBlocks
                        );
                        return;
                    }

                    if (tools.containsKey(action)) {
                        Map<String, String> toolParams = constructToolParams(
                            tools,
                            toolSpecMap,
                            question,
                            lastActionInput,
                            action,
                            actionInput
                        );
                        lastToolParams.clear();
                        lastToolParams.putAll(toolParams);
                        runTool(
                            tools,
                            toolSpecMap,
                            tmpParameters,
                            (ActionListener<Object>) nextStepListener,
                            action,
                            actionInput,
                            toolParams,
                            interactions,
                            toolCallId,
                            functionCalling
                        );

                    } else {
                        String res = String.format(Locale.ROOT, "Failed to run the tool %s which is unsupported.", action);
                        StringSubstitutor substitutor = new StringSubstitutor(
                            Map.of(SCRATCHPAD, scratchpadBuilder.toString()),
                            "${parameters.",
                            "}"
                        );
                        newPrompt.set(substitutor.replace(finalPrompt));
                        tmpParameters.put(PROMPT, newPrompt.get());
                        ((ActionListener<Object>) nextStepListener).onResponse(res);
                    }
                } else {
                    Object filteredOutput = filterToolOutput(lastToolParams, output);
                    addToolOutputToAddtionalInfo(toolSpecMap, lastAction, additionalInfo, filteredOutput);

                    String toolResponse = constructToolResponse(
                        tmpParameters,
                        lastAction,
                        lastActionInput,
                        lastToolSelectionResponse,
                        filteredOutput
                    );
                    scratchpadBuilder.append(toolResponse).append("\n\n");

                    saveTraceData(
                        memory,
                        "ReAct",
                        lastActionInput.get(),
                        outputToOutputString(filteredOutput),
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        lastAction.get(),
                        inputContentBlocks
                    );

                    StringSubstitutor substitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder), "${parameters.", "}");
                    newPrompt.set(substitutor.replace(finalPrompt));
                    tmpParameters.put(PROMPT, newPrompt.get());
                    if (!interactions.isEmpty()) {
                        tmpParameters.put(INTERACTIONS, ", " + String.join(", ", interactions));
                    }

                    sessionMsgAnswerBuilder.append(outputToOutputString(filteredOutput));
                    streamingWrapper.sendToolResponse(outputToOutputString(output), sessionId, parentInteractionId);
                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(
                                    Collections
                                        .singletonList(
                                            ModelTensor.builder().name("response").result(sessionMsgAnswerBuilder.toString()).build()
                                        )
                                )
                                .build()
                        );

                    if (finalI == maxIterations - 1) {
                        handleMaxIterationsReached(
                            sessionId,
                            listener,
                            question,
                            parentInteractionId,
                            verbose,
                            traceDisabled,
                            traceTensors,
                            memory,
                            traceNumber,
                            additionalInfo,
                            lastThought,
                            maxIterations,
                            tools,
                            inputContentBlocks
                        );
                        return;
                    }
                    ActionRequest request = streamingWrapper.createPredictionRequest(llm, tmpParameters, tenantId);
                    streamingWrapper.executeRequest(request, (ActionListener<MLTaskResponse>) nextStepListener);
                }
            }, e -> {
                log.error("Failed to run chat agent", e);
                listener.onFailure(e);
            });
            if (nextStepListener != null) {
                lastStepListener = nextStepListener;
            }
        }

        ActionRequest request = streamingWrapper.createPredictionRequest(llm, tmpParameters, tenantId);
        streamingWrapper.executeRequest(request, firstListener);
    }

    private static List<ModelTensors> createFinalAnswerTensors(List<ModelTensors> sessionId, List<ModelTensor> lastThought) {
        List<ModelTensors> finalModelTensors = sessionId;
        finalModelTensors.add(ModelTensors.builder().mlModelTensors(lastThought).build());
        return finalModelTensors;
    }

    private static String constructToolResponse(
        Map<String, String> tmpParameters,
        AtomicReference<String> lastAction,
        AtomicReference<String> lastActionInput,
        AtomicReference<String> lastToolSelectionResponse,
        Object output
    ) throws PrivilegedActionException {
        String toolResponse = tmpParameters.get(TOOL_RESPONSE);
        StringSubstitutor toolResponseSubstitutor = new StringSubstitutor(
            Map
                .of(
                    "llm_tool_selection_response",
                    lastToolSelectionResponse.get(),
                    "tool_name",
                    lastAction.get(),
                    "tool_input",
                    lastActionInput.get(),
                    "observation",
                    outputToOutputString(output)
                ),
            "${parameters.",
            "}"
        );
        toolResponse = toolResponseSubstitutor.replace(toolResponse);
        return toolResponse;
    }

    private static void addToolOutputToAddtionalInfo(
        Map<String, MLToolSpec> toolSpecMap,
        AtomicReference<String> lastAction,
        Map<String, Object> additionalInfo,
        Object output
    ) throws PrivilegedActionException {
        MLToolSpec toolSpec = toolSpecMap.get(lastAction.get());
        if (toolSpec != null && toolSpec.isIncludeOutputInAgentResponse()) {
            String outputString = outputToOutputString(output);
            String toolOutputKey = String.format("%s.output", getToolName(toolSpec));
            if (additionalInfo.get(toolOutputKey) != null) {
                List<String> list = (List<String>) additionalInfo.get(toolOutputKey);
                list.add(outputString);
            } else {
                additionalInfo.put(toolOutputKey, Lists.newArrayList(outputString));
            }
        }
    }

    private static void runTool(
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> tmpParameters,
        ActionListener<Object> nextStepListener,
        String action,
        String actionInput,
        Map<String, String> toolParams,
        List<String> interactions,
        String toolCallId,
        FunctionCalling functionCalling
    ) {
        if (tools.get(action).validate(toolParams)) {
            try {
                String finalAction = action;
                ActionListener<Object> toolListener = ActionListener.wrap(r -> {
                    if (functionCalling != null) {
                        String outputResponse = parseResponse(filterToolOutput(toolParams, r));
                        List<Map<String, Object>> toolResults = List
                            .of(Map.of(TOOL_CALL_ID, toolCallId, TOOL_RESULT, Map.of("text", outputResponse)));
                        List<LLMMessage> llmMessages = functionCalling.supply(toolResults);
                        // TODO: support multiple tool calls at the same time so that multiple LLMMessages can be generated here
                        interactions.add(llmMessages.getFirst().getResponse());
                    } else {
                        interactions
                            .add(
                                substitute(
                                    tmpParameters.get(INTERACTION_TEMPLATE_TOOL_RESPONSE),
                                    Map.of(TOOL_CALL_ID, toolCallId, "tool_response", processTextDoc(StringUtils.toJson(r))),
                                    INTERACTIONS_PREFIX
                                )
                            );
                    }
                    nextStepListener.onResponse(r);
                }, e -> {
                    interactions
                        .add(
                            substitute(
                                tmpParameters.get(INTERACTION_TEMPLATE_TOOL_RESPONSE),
                                Map.of(TOOL_CALL_ID, toolCallId, "tool_response", "Tool " + action + " failed: " + e.getMessage()),
                                INTERACTIONS_PREFIX
                            )
                        );
                    nextStepListener
                        .onResponse(
                            String
                                .format(
                                    Locale.ROOT,
                                    "Failed to run the tool %s with the error message %s.",
                                    finalAction,
                                    e.getMessage().replaceAll("\\n", "\n")
                                )
                        );
                });
                if (tools.get(action) instanceof MLModelTool) {
                    Map<String, String> llmToolTmpParameters = new HashMap<>();
                    llmToolTmpParameters.putAll(tmpParameters);
                    llmToolTmpParameters.putAll(toolSpecMap.get(action).getParameters());
                    llmToolTmpParameters.put(MLAgentExecutor.QUESTION, actionInput);
                    tools.get(action).run(llmToolTmpParameters, toolListener); // run tool
                    updateParametersAcrossTools(tmpParameters, llmToolTmpParameters);
                } else {
                    Map<String, String> parameters = new HashMap<>();
                    parameters.putAll(tmpParameters);
                    parameters.putAll(toolParams);
                    tools.get(action).run(parameters, toolListener); // run tool
                    updateParametersAcrossTools(tmpParameters, parameters);
                }
            } catch (Exception e) {
                log.error("Failed to run tool {}", action, e);
                nextStepListener
                    .onResponse(String.format(Locale.ROOT, "Failed to run the tool %s with the error message %s.", action, e.getMessage()));
            }
        } else { // TODO: add failure to interaction to let LLM regenerate ?
            String res = String.format(Locale.ROOT, "Failed to run the tool %s due to wrong input %s.", action, actionInput);
            nextStepListener.onResponse(res);
        }
    }

    /**
     * In each tool runs, it copies agent parameters, which is tmpParameters into a new set of parameter llmToolTmpParameters,
     * after the tool runs, normally llmToolTmpParameters will be discarded, but for some special parameters like SCRATCHPAD_NOTES_KEY,
     * some new llmToolTmpParameters produced by the tool run can opt to be copied back to tmpParameters to share across tools in the same interaction
     * @param tmpParameters
     * @param llmToolTmpParameters
     */
    private static void updateParametersAcrossTools(Map<String, String> tmpParameters, Map<String, String> llmToolTmpParameters) {
        // update the tmpParameters if the tool run produce new scratch pad
        if (llmToolTmpParameters.containsKey(SCRATCHPAD_NOTES_KEY) && llmToolTmpParameters.get(SCRATCHPAD_NOTES_KEY) != "[]") {
            tmpParameters.put(SCRATCHPAD_NOTES_KEY, llmToolTmpParameters.getOrDefault(SCRATCHPAD_NOTES_KEY, "[]"));
        }
    }

    // Todo: can we use bedrock converse model provider for the same
    /**
     * Converts content blocks to Bedrock Converse API format for chat history.
     * This reuses the same conversion logic as BedrockConverseModelProvider.
     */
    private static String convertContentBlocksToBedrockFormat(List<org.opensearch.ml.common.agent.ContentBlock> contentBlocks) {
        if (contentBlocks == null || contentBlocks.isEmpty()) {
            return null;
        }

        StringBuilder contentArray = new StringBuilder();
        boolean first = true;
        for (org.opensearch.ml.common.agent.ContentBlock block : contentBlocks) {
            if (!first) {
                contentArray.append(",");
            }
            first = false;

            switch (block.getType()) {
                case TEXT:
                    contentArray.append("{\"text\":\"").append(StringEscapeUtils.escapeJson(block.getText())).append("\"}");
                    break;
                case IMAGE:
                    if (block.getImage() != null) {
                        String sourceType = block.getImage().getType() == org.opensearch.ml.common.agent.SourceType.URL
                            ? "s3Location"
                            : "bytes";
                        contentArray
                            .append("{\"image\":{\"format\":\"")
                            .append(block.getImage().getFormat())
                            .append("\",\"source\":{\"")
                            .append(sourceType)
                            .append("\":\"")
                            .append(block.getImage().getData())
                            .append("\"}}}");
                    }
                    break;
                case VIDEO:
                    if (block.getVideo() != null) {
                        String sourceType = block.getVideo().getType() == org.opensearch.ml.common.agent.SourceType.URL
                            ? "s3Location"
                            : "bytes";
                        contentArray
                            .append("{\"video\":{\"format\":\"")
                            .append(block.getVideo().getFormat())
                            .append("\",\"source\":{\"")
                            .append(sourceType)
                            .append("\":\"")
                            .append(block.getVideo().getData())
                            .append("\"}}}");
                    }
                    break;
                case DOCUMENT:
                    if (block.getDocument() != null) {
                        String sourceType = block.getDocument().getType() == org.opensearch.ml.common.agent.SourceType.URL
                            ? "s3Location"
                            : "bytes";
                        contentArray
                            .append("{\"document\":{\"format\":\"")
                            .append(block.getDocument().getFormat())
                            .append("\",\"name\":\"document\",\"source\":{\"")
                            .append(sourceType)
                            .append("\":\"")
                            .append(block.getDocument().getData())
                            .append("\"}}}");
                    }
                    break;
                default:
                    // Skip unsupported content types
                    break;
            }
        }

        return "[" + contentArray.toString() + "]";
    }

    public static void saveTraceData(
        Memory memory,
        String memoryType,
        String question,
        String thoughtResponse,
        String sessionId,
        boolean traceDisabled,
        String parentInteractionId,
        AtomicInteger traceNumber,
        String origin
    ) {
        if (memory != null) {
            ConversationIndexMessage msgTemp = ConversationIndexMessage
                .conversationIndexMessageBuilder()
                .type(memoryType)
                .question(question)
                .response(thoughtResponse)
                .finalAnswer(false)
                .sessionId(sessionId)
                .build();
            if (!traceDisabled) {
                memory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), origin);
            }
        }
    }

    /**
     * Enhanced saveTraceData method that supports multi-modal input content.
     * This method accepts input content blocks for storing rich media in memory.
     */
    public static void saveTraceData(
        Memory memory,
        String memoryType,
        String question,
        String thoughtResponse,
        String sessionId,
        boolean traceDisabled,
        String parentInteractionId,
        AtomicInteger traceNumber,
        String origin,
        List<org.opensearch.ml.common.agent.ContentBlock> inputContentBlocks
    ) {
        if (memory != null) {
            ConversationIndexMessage msgTemp;
            if (inputContentBlocks != null && !inputContentBlocks.isEmpty()) {
                msgTemp = ConversationIndexMessage
                    .conversationIndexMessageBuilderWithContent()
                    .type(memoryType)
                    .question(question)
                    .response(thoughtResponse)
                    .finalAnswer(false)
                    .sessionId(sessionId)
                    .inputContentBlocks(inputContentBlocks)
                    .build();
            } else {
                msgTemp = ConversationIndexMessage
                    .conversationIndexMessageBuilder()
                    .type(memoryType)
                    .question(question)
                    .response(thoughtResponse)
                    .finalAnswer(false)
                    .sessionId(sessionId)
                    .build();
            }
            if (!traceDisabled) {
                memory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), origin);
            }
        }
    }

    private void sendFinalAnswer(
        String sessionId,
        ActionListener<Object> listener,
        String question,
        String parentInteractionId,
        boolean verbose,
        boolean traceDisabled,
        List<ModelTensors> cotModelTensors,
        Memory memory,
        AtomicInteger traceNumber,
        Map<String, Object> additionalInfo,
        String finalAnswer,
        List<ContentBlock> inputContentBlocks
    ) {
        // Send completion chunk for streaming
        streamingWrapper.sendCompletionChunk(sessionId, parentInteractionId);

        if (memory != null) {
            String copyOfFinalAnswer = finalAnswer;
            ActionListener saveTraceListener = ActionListener.wrap(r -> {
                memory
                    .update(
                        parentInteractionId,
                        Map.of(AI_RESPONSE_FIELD, copyOfFinalAnswer, ADDITIONAL_INFO_FIELD, additionalInfo),
                        ActionListener.wrap(res -> {
                            returnFinalResponse(
                                sessionId,
                                listener,
                                parentInteractionId,
                                verbose,
                                cotModelTensors,
                                additionalInfo,
                                copyOfFinalAnswer
                            );
                        }, e -> { listener.onFailure(e); })
                    );
            }, e -> { listener.onFailure(e); });

            // todo: overload this method
            if (inputContentBlocks != null && !inputContentBlocks.isEmpty()) {
                saveMessageWithContent(
                    memory,
                    question,
                    finalAnswer,
                    sessionId,
                    parentInteractionId,
                    traceNumber,
                    true,
                    traceDisabled,
                    saveTraceListener,
                    inputContentBlocks
                );
            } else {
                saveMessage(
                    memory,
                    question,
                    finalAnswer,
                    sessionId,
                    parentInteractionId,
                    traceNumber,
                    true,
                    traceDisabled,
                    saveTraceListener
                );
            }
        } else {
            streamingWrapper
                .sendFinalResponse(sessionId, listener, parentInteractionId, verbose, cotModelTensors, additionalInfo, finalAnswer);
        }
    }

    public static List<ModelTensors> createModelTensors(String sessionId, String parentInteractionId) {
        List<ModelTensors> cotModelTensors = new ArrayList<>();

        cotModelTensors
            .add(
                ModelTensors
                    .builder()
                    .mlModelTensors(
                        List
                            .of(
                                ModelTensor.builder().name(MLAgentExecutor.MEMORY_ID).result(sessionId).build(),
                                ModelTensor.builder().name(MLAgentExecutor.PARENT_INTERACTION_ID).result(parentInteractionId).build()
                            )
                    )
                    .build()
            );
        return cotModelTensors;
    }

    private static String constructLLMPrompt(Map<String, Tool> tools, Map<String, String> tmpParameters) {
        String prompt = tmpParameters.getOrDefault(PROMPT, PromptTemplate.PROMPT_TEMPLATE);
        StringSubstitutor promptSubstitutor = new StringSubstitutor(tmpParameters, "${parameters.", "}");
        prompt = promptSubstitutor.replace(prompt);
        prompt = AgentUtils.addPrefixSuffixToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addToolsToPrompt(tools, tmpParameters, getToolNames(tools), prompt);
        prompt = AgentUtils.addIndicesToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addExamplesToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addChatHistoryToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addContextToPrompt(tmpParameters, prompt);
        return prompt;
    }

    @VisibleForTesting
    static Map<String, String> constructLLMParams(LLMSpec llm, Map<String, String> parameters) {
        Map<String, String> tmpParameters = new HashMap<>();
        if (llm.getParameters() != null) {
            tmpParameters.putAll(llm.getParameters());
        }
        tmpParameters.putAll(parameters);
        if (!tmpParameters.containsKey("stop")) {
            tmpParameters.put("stop", gson.toJson(new String[] { "\nObservation:", "\n\tObservation:" }));
        }
        if (!tmpParameters.containsKey("stop_sequences")) {
            tmpParameters
                .put(
                    "stop_sequences",
                    gson
                        .toJson(
                            new String[] {
                                "\n\nHuman:",
                                "\nObservation:",
                                "\n\tObservation:",
                                "\nObservation",
                                "\n\tObservation",
                                "\n\nQuestion" }
                        )
                );
        }

        boolean injectDate = Boolean.parseBoolean(tmpParameters.getOrDefault(INJECT_DATETIME_FIELD, "false"));
        if (injectDate) {
            String dateFormat = tmpParameters.get(DATETIME_FORMAT_FIELD);
            String currentDateTime = getCurrentDateTime(dateFormat);
            // If system_prompt exists, inject datetime into it
            if (tmpParameters.containsKey(SYSTEM_PROMPT_FIELD)) {
                String systemPrompt = tmpParameters.get(SYSTEM_PROMPT_FIELD);
                systemPrompt = systemPrompt + "\n\n" + currentDateTime;
                tmpParameters.put(SYSTEM_PROMPT_FIELD, systemPrompt);
            } else {
                // Otherwise inject datetime into prompt_prefix
                String promptPrefix = tmpParameters.getOrDefault(PROMPT_PREFIX, PromptTemplate.PROMPT_TEMPLATE_PREFIX);
                promptPrefix = promptPrefix + "\n\n" + currentDateTime;
                tmpParameters.put(PROMPT_PREFIX, promptPrefix);
            }
        }

        tmpParameters.putIfAbsent(PROMPT_PREFIX, PromptTemplate.PROMPT_TEMPLATE_PREFIX);
        tmpParameters.putIfAbsent(PROMPT_SUFFIX, PromptTemplate.PROMPT_TEMPLATE_SUFFIX);
        tmpParameters.putIfAbsent(RESPONSE_FORMAT_INSTRUCTION, PromptTemplate.PROMPT_FORMAT_INSTRUCTION);
        tmpParameters.putIfAbsent(TOOL_RESPONSE, PromptTemplate.PROMPT_TEMPLATE_TOOL_RESPONSE);

        // Set default system prompt only if none exists
        if (!tmpParameters.containsKey(SYSTEM_PROMPT_FIELD)) {
            String systemPrompt = DEFAULT_SYSTEM_PROMPT;
            // If datetime injection was enabled, include it in the default system prompt
            if (injectDate) {
                String dateFormat = tmpParameters.get(DATETIME_FORMAT_FIELD);
                String currentDateTime = getCurrentDateTime(dateFormat);
                systemPrompt = systemPrompt + "\n\n" + currentDateTime;
            }
            tmpParameters.put(SYSTEM_PROMPT_FIELD, systemPrompt);
        }

        return tmpParameters;
    }

    public static void returnFinalResponse(
        String sessionId,
        ActionListener<Object> listener,
        String parentInteractionId,
        boolean verbose,
        List<ModelTensors> cotModelTensors, // AtomicBoolean getFinalAnswer,
        Map<String, Object> additionalInfo,
        String finalAnswer2
    ) {
        cotModelTensors
            .add(
                ModelTensors.builder().mlModelTensors(List.of(ModelTensor.builder().name("response").result(finalAnswer2).build())).build()
            );

        List<ModelTensors> finalModelTensors = createFinalAnswerTensors(
            createModelTensors(sessionId, parentInteractionId),
            List
                .of(
                    ModelTensor
                        .builder()
                        .name("response")
                        .dataAsMap(ImmutableMap.of("response", finalAnswer2, ADDITIONAL_INFO_FIELD, additionalInfo))
                        .build()
                )
        );
        if (verbose) {
            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build());
        } else {
            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build());
        }
    }

    private void handleMaxIterationsReached(
        String sessionId,
        ActionListener<Object> listener,
        String question,
        String parentInteractionId,
        boolean verbose,
        boolean traceDisabled,
        List<ModelTensors> traceTensors,
        Memory memory,
        AtomicInteger traceNumber,
        Map<String, Object> additionalInfo,
        AtomicReference<String> lastThought,
        int maxIterations,
        Map<String, Tool> tools,
        List<ContentBlock> inputContentBlocks
    ) {
        String incompleteResponse = (lastThought.get() != null && !lastThought.get().isEmpty() && !"null".equals(lastThought.get()))
            ? String.format("%s. Last thought: %s", String.format(MAX_ITERATIONS_MESSAGE, maxIterations), lastThought.get())
            : String.format(MAX_ITERATIONS_MESSAGE, maxIterations);
        sendFinalAnswer(
            sessionId,
            listener,
            question,
            parentInteractionId,
            verbose,
            traceDisabled,
            traceTensors,
            memory,
            traceNumber,
            additionalInfo,
            incompleteResponse,
            inputContentBlocks
        );
        cleanUpResource(tools);
    }

    private void saveMessage(
        Memory memory,
        String question,
        String finalAnswer,
        String sessionId,
        String parentInteractionId,
        AtomicInteger traceNumber,
        boolean isFinalAnswer,
        boolean traceDisabled,
        ActionListener listener
    ) {
        ConversationIndexMessage msgTemp = ConversationIndexMessage
            .conversationIndexMessageBuilder()
            .type(memory.getType())
            .question(question)
            .response(finalAnswer)
            .finalAnswer(isFinalAnswer)
            .sessionId(sessionId)
            .build();
        if (traceDisabled) {
            listener.onResponse(true);
        } else {
            memory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), "LLM", listener);
        }
    }

    /**
     * Enhanced saveMessage method that supports multi-modal input content.
     * This method accepts input content blocks for storing rich media in memory.
     */
    private void saveMessageWithContent(
        Memory memory,
        String question,
        String finalAnswer,
        String sessionId,
        String parentInteractionId,
        AtomicInteger traceNumber,
        boolean isFinalAnswer,
        boolean traceDisabled,
        ActionListener listener,
        List<org.opensearch.ml.common.agent.ContentBlock> inputContentBlocks
    ) {
        ConversationIndexMessage msgTemp = ConversationIndexMessage
            .conversationIndexMessageBuilderWithContent()
            .type(memory.getType())
            .question(question)
            .response(finalAnswer)
            .finalAnswer(isFinalAnswer)
            .sessionId(sessionId)
            .inputContentBlocks(inputContentBlocks)
            .build();
        if (traceDisabled) {
            listener.onResponse(true);
        } else {
            memory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), "LLM", listener);
        }
    }
}
