/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_FINISH_REASON_PATH;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_FINISH_REASON_TOOL_USE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_GEN_INPUT;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_RESPONSE_FILTER;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_SUFFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALLS_PATH;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALLS_TOOL_INPUT;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALLS_TOOL_NAME;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALL_ID;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALL_ID_PATH;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.ACTION;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.ACTION_INPUT;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.CHAT_HISTORY;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.CONTEXT;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.EXAMPLES;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.FINAL_ANSWER;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.OS_INDICES;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.THOUGHT;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.THOUGHT_RESPONSE;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Tool;

public class AgentUtilsTest {

    @Mock
    private Tool tool1, tool2;

    private Map<String, Map<String, String>> llmResponseExpectedParseResults;

    private String responseForAction = "---------------------\n{\n  "
        + "\"thought\": \"Let me search our index to find population projections\", \n  "
        + "\"action\": \"VectorDBTool\",\n  "
        + "\"action_input\": \"Seattle population projection 2023\"\n}";

    private String responseForActionWrongAction = "---------------------\n{\n  "
        + "\"thought\": \"Let me search our index to find population projections\", \n  "
        + "\"action\": \"Let me run VectorDBTool to get more data\",\n  "
        + "\"action_input\": \"Seattle population projection 2023\"\n}";

    private String responseForActionNullAction = "---------------------\n{\n  "
        + "\"thought\": \"Let me search our index to find population projections\" \n  }";

    private String responseNotFollowJsonFormat = "Final answer is I don't know";
    private String responseForActionInvalidJson = "---------------------\n{\n  "
        + "\"thought\": \"Let me search our index to find population projections\", \n  "
        + "\"action\": \"VectorDBTool\",\n  "
        + "\"action_input\": \"Seattle population projection 2023\"";
    private String responseForFinalAnswer = "---------------------```json\n{\n  "
        + "\"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n  "
        + "\"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}\n```";
    private String responseForFinalAnswerInvalidJson =
        "\"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n  "
            + "\"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}\n```";

    private String responseForFinalAnswerWithJson = "---------------------```json\n{\n  "
        + "\"thought\": \"Now I know the final answer\",\n  "
        + "\"final_answer\": \"PPLTool generates such query ```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans centroids=3 ```.\"\n}\n```";

    private String responseForFinalAnswerWithMultilines = "---------------------```json\n{\n  "
        + "\"thought\": \"Now I know the final answer\",\n  "
        + "\"final_answer\": \"PPLTool generates such query \n```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans centroids=3 ```.\"\n}\n```";

    private String responseForFinalAnswerWithQuotes = "---------------------```json\n{\n  "
        + "\"thought\": \"Now I know the final answer\",\n  "
        + "\"final_answer\": \"PPLTool generates such query \n```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans name=\"Jack\" ```.\"\n}\n```";

    private String wrongResponseForAction = "---------------------```json\n{\n  "
        + "\"thought\": \"Let's try VectorDBTool\",\n  "
        + "\"action\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}\n```";

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        llmResponseExpectedParseResults = new HashMap<>();
        Map responseForActionExpectedResult = Map
            .of(
                THOUGHT,
                "Let me search our index to find population projections",
                ACTION,
                "VectorDBTool",
                ACTION_INPUT,
                "Seattle population projection 2023"
            );
        llmResponseExpectedParseResults.put(responseForAction, responseForActionExpectedResult);
        llmResponseExpectedParseResults.put(responseForActionWrongAction, responseForActionExpectedResult);
        llmResponseExpectedParseResults.put(responseForActionInvalidJson, responseForActionExpectedResult);
        Map responseForActionNullActionExpectedResult = Map
            .of(
                THOUGHT,
                "Let me search our index to find population projections",
                FINAL_ANSWER,
                "{\n  \"thought\": \"Let me search our index to find population projections\" \n  }"
            );
        llmResponseExpectedParseResults.put(responseForActionNullAction, responseForActionNullActionExpectedResult);

        Map responseNotFollowJsonFormatExpectedResult = Map.of(FINAL_ANSWER, responseNotFollowJsonFormat);
        llmResponseExpectedParseResults.put(responseNotFollowJsonFormat, responseNotFollowJsonFormatExpectedResult);

        Map responseForFinalAnswerExpectedResult = Map
            .of(
                THOUGHT,
                "Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:",
                FINAL_ANSWER,
                "After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius."
            );
        llmResponseExpectedParseResults.put(responseForFinalAnswer, responseForFinalAnswerExpectedResult);
        Map responseForFinalAnswerExpectedResultExpectedResult = Map
            .of(
                THOUGHT,
                "Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:",
                FINAL_ANSWER,
                "After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius."
            );
        llmResponseExpectedParseResults.put(responseForFinalAnswerInvalidJson, responseForFinalAnswerExpectedResultExpectedResult);
        Map responseForFinalAnswerWithJsonExpectedResultExpectedResult = Map
            .of(
                THOUGHT,
                "Now I know the final answer",
                FINAL_ANSWER,
                "PPLTool generates such query ```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans centroids=3 ```."
            );
        llmResponseExpectedParseResults.put(responseForFinalAnswerWithJson, responseForFinalAnswerWithJsonExpectedResultExpectedResult);

        Map wrongResponseForActionExpectedResultExpectedResult = Map
            .of(
                THOUGHT,
                "Let's try VectorDBTool",
                FINAL_ANSWER,
                "{\n"
                    + "  \"thought\": \"Let's try VectorDBTool\",\n"
                    + "  \"action\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n"
                    + "}"
            );
        llmResponseExpectedParseResults.put(wrongResponseForAction, wrongResponseForActionExpectedResultExpectedResult);

        Map responseForFinalAnswerWithMultilinesExpectedResult = Map
            .of(
                THOUGHT,
                "Now I know the final answer",
                FINAL_ANSWER,
                "PPLTool generates such query \n```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans centroids=3 ```."
            );
        llmResponseExpectedParseResults.put(responseForFinalAnswerWithMultilines, responseForFinalAnswerWithMultilinesExpectedResult);

        Map responseForFinalAnswerWithQuotesExpectedResult = Map
            .of(
                THOUGHT,
                "Now I know the final answer",
                FINAL_ANSWER,
                "PPLTool generates such query \n```json source=iris_data | fields petal_length_in_cm,petal_width_in_cm | kmeans name=\"Jack\" ```."
            );
        llmResponseExpectedParseResults.put(responseForFinalAnswerWithQuotes, responseForFinalAnswerWithQuotesExpectedResult);

    }

    @Test
    public void testAddIndicesToPrompt_WithIndices() {
        String initialPrompt = "initial prompt ${parameters.opensearch_indices}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(OS_INDICES, "[\"index1\", \"index2\"]");

        String expected =
            "initial prompt You have access to the following OpenSearch Index defined in <opensearch_indexes>: \n<opensearch_indexes>\n"
                + "<index>\nindex1\n</index>\n<index>\nindex2\n</index>\n</opensearch_indexes>\n";

        String result = AgentUtils.addIndicesToPrompt(parameters, initialPrompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddIndicesToPrompt_WithoutIndices() {
        String prompt = "initial prompt";
        Map<String, String> parameters = new HashMap<>();

        String expected = "initial prompt";

        String result = AgentUtils.addIndicesToPrompt(parameters, prompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddIndicesToPrompt_WithCustomPrefixSuffix() {
        String initialPrompt = "initial prompt ${parameters.opensearch_indices}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(OS_INDICES, "[\"index1\", \"index2\"]");
        parameters.put("opensearch_indices.prefix", "Custom Prefix\n");
        parameters.put("opensearch_indices.suffix", "\nCustom Suffix");
        parameters.put("opensearch_indices.index.prefix", "Index: ");
        parameters.put("opensearch_indices.index.suffix", "; ");

        String expected = "initial prompt Custom Prefix\nIndex: index1; Index: index2; \nCustom Suffix";

        String result = AgentUtils.addIndicesToPrompt(parameters, initialPrompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddExamplesToPrompt_WithExamples() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.examples}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(EXAMPLES, "[\"Example 1\", \"Example 2\"]");

        // Expected output
        String expectedPrompt = "initial prompt EXAMPLES\n--------\n"
            + "You should follow and learn from examples defined in <examples>: \n"
            + "<examples>\n"
            + "<example>\nExample 1\n</example>\n"
            + "<example>\nExample 2\n</example>\n"
            + "</examples>\n";

        // Call the method under test
        String actualPrompt = AgentUtils.addExamplesToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddExamplesToPrompt_WithoutExamples() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.examples}";
        Map<String, String> parameters = new HashMap<>();

        // Expected output (should remain unchanged)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addExamplesToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddPrefixSuffixToPrompt_WithPrefixSuffix() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.prompt.prefix} main content ${parameters.prompt.suffix}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(PROMPT_PREFIX, "Prefix: ");
        parameters.put(PROMPT_SUFFIX, " :Suffix");

        // Expected output
        String expectedPrompt = "initial prompt Prefix:  main content  :Suffix";

        // Call the method under test
        String actualPrompt = AgentUtils.addPrefixSuffixToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddPrefixSuffixToPrompt_WithoutPrefixSuffix() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.prompt.prefix} main content ${parameters.prompt.suffix}";
        Map<String, String> parameters = new HashMap<>();

        // Expected output (should remain unchanged)
        String expectedPrompt = "initial prompt  main content ";

        // Call the method under test
        String actualPrompt = AgentUtils.addPrefixSuffixToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddToolsToPrompt_WithDescriptions() {
        // Setup
        Map<String, Tool> tools = new HashMap<>();
        tools.put("Tool1", tool1);
        tools.put("Tool2", tool2);
        when(tool1.getDescription()).thenReturn("Description of Tool1");
        when(tool2.getDescription()).thenReturn("Description of Tool2");

        List<String> inputTools = Arrays.asList("Tool1", "Tool2");
        String initialPrompt = "initial prompt ${parameters.tool_descriptions} and ${parameters.tool_names}";

        // Expected output
        String expectedPrompt = "initial prompt You have access to the following tools defined in <tools>: \n"
            + "<tools>\n<tool>\nTool1: Description of Tool1\n</tool>\n"
            + "<tool>\nTool2: Description of Tool2\n</tool>\n</tools>\n and Tool1, Tool2,";

        // Call the method under test
        String actualPrompt = AgentUtils.addToolsToPrompt(tools, new HashMap<>(), inputTools, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddToolsToPrompt_ToolNotRegistered() {
        // Setup
        Map<String, Tool> tools = new HashMap<>();
        tools.put("Tool1", tool1);
        List<String> inputTools = Arrays.asList("Tool1", "UnregisteredTool");
        String initialPrompt = "initial prompt ${parameters.tool_descriptions}";

        // Assert
        assertThrows(IllegalArgumentException.class, () -> AgentUtils.addToolsToPrompt(tools, new HashMap<>(), inputTools, initialPrompt));
    }

    @Test
    public void testAddChatHistoryToPrompt_WithChatHistory() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        parameters.put(CHAT_HISTORY, "Previous chat history here.");
        String initialPrompt = "initial prompt ${parameters.chat_history}";

        // Expected output
        String expectedPrompt = "initial prompt Previous chat history here.";

        // Call the method under test
        String actualPrompt = AgentUtils.addChatHistoryToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddChatHistoryToPrompt_NoChatHistory() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        String initialPrompt = "initial prompt ${parameters.chat_history}";

        // Expected output (no change from initial prompt)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addChatHistoryToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddContextToPrompt_WithContext() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        parameters.put(CONTEXT, "Contextual information here.");
        String initialPrompt = "initial prompt ${parameters.context}";

        // Expected output
        String expectedPrompt = "initial prompt Contextual information here.";

        // Call the method under test
        String actualPrompt = AgentUtils.addContextToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddContextToPrompt_NoContext() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        String initialPrompt = "initial prompt ${parameters.context}";

        // Expected output (no change from initial prompt)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addContextToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testExtractModelResponseJsonWithInvalidModelOutput() {
        String text = "invalid output";
        assertThrows(IllegalArgumentException.class, () -> AgentUtils.extractModelResponseJson(text));
    }

    @Test
    public void testExtractModelResponseJsonWithValidModelOutput() {
        String text =
            "This is the model response\n```json\n{\"thought\":\"use ListIndexTool to get index first\",\"action\":\"ListIndexTool\"} \n``` other content";
        String responseJson = AgentUtils.extractModelResponseJson(text);
        assertEquals("{\"thought\":\"use ListIndexTool to get index first\",\"action\":\"ListIndexTool\"}", responseJson);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtFinalAnswer() {
        String text =
            "---------------------\n{\n  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}";
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n"
            + "  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n"
            + "}";
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtFinalAnswerJsonBlock() {
        String text = responseForFinalAnswer;
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n"
            + "  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n"
            + "}";
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtActionInput() {
        String text = responseForAction;
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Let me search our index to find population projections\", \n"
            + "  \"action\": \"VectorDBTool\",\n"
            + "  \"action_input\": \"Seattle population projection 2023\"\n"
            + "}";
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void testExtractMethods() {
        List<String> textList = List.of(responseForAction, responseForActionInvalidJson);
        for (String text : textList) {
            String thought = AgentUtils.extractThought(text);
            String action = AgentUtils.extractAction(text);
            String actionInput = AgentUtils.extractActionInput(text);
            String finalAnswer = AgentUtils.extractFinalAnswer(text);
            Assert.assertEquals("Let me search our index to find population projections", thought);
            Assert.assertEquals("VectorDBTool\",\n  ", action);
            Assert.assertEquals("Seattle population projection 2023", actionInput);
            Assert.assertNull(finalAnswer);
        }
    }

    @Test
    public void testExtractMethods_FinalAnswer() {
        List<String> textList = List.of(responseForFinalAnswer, responseForFinalAnswerInvalidJson);
        for (String text : textList) {
            String thought = AgentUtils.extractThought(text);
            String action = AgentUtils.extractAction(text);
            String actionInput = AgentUtils.extractActionInput(text);
            String finalAnswer = AgentUtils.extractFinalAnswer(text);
            Assert
                .assertEquals(
                    "Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:",
                    thought
                );
            Assert.assertNull(action);
            Assert.assertNull(actionInput);
            Assert
                .assertEquals(
                    "After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.",
                    finalAnswer
                );
        }
    }

    @Test
    public void testParseLLMOutput() {
        Set<String> tools = Set.of("VectorDBTool", "ListIndexTool");
        for (Map.Entry<String, Map<String, String>> entry : llmResponseExpectedParseResults.entrySet()) {
            ModelTensorOutput modelTensoOutput = ModelTensorOutput
                .builder()
                .mlModelOutputs(
                    List
                        .of(
                            ModelTensors
                                .builder()
                                .mlModelTensors(
                                    List.of(ModelTensor.builder().name("response").dataAsMap(Map.of("response", entry.getKey())).build())
                                )
                                .build()
                        )
                )
                .build();
            Map<String, String> output = AgentUtils
                .parseLLMOutput(Collections.emptyMap(), modelTensoOutput, null, tools, Collections.emptyList());
            for (String key : entry.getValue().keySet()) {
                Assert.assertEquals(entry.getValue().get(key), output.get(key));
            }
        }
    }

    @Test
    public void testParseLLMOutput_MultipleFields() {
        Set<String> tools = Set.of("VectorDBTool", "ListIndexTool");
        String thought = "Let me run VectorDBTool to get more information";
        String toolName = "vectordbtool";
        ModelTensorOutput modelTensoOutput = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(
                                List
                                    .of(
                                        ModelTensor.builder().name("response").dataAsMap(Map.of(THOUGHT, thought, ACTION, toolName)).build()
                                    )
                            )
                            .build()
                    )
            )
            .build();
        Map<String, String> output = AgentUtils
            .parseLLMOutput(Collections.emptyMap(), modelTensoOutput, null, tools, Collections.emptyList());
        Assert.assertEquals(3, output.size());
        Assert.assertEquals(thought, output.get(THOUGHT));
        Assert.assertEquals("VectorDBTool", output.get(ACTION));
        Set<String> expected = Set
            .of(
                "{\"action\":\"vectordbtool\",\"thought\":\"Let me run VectorDBTool to get more information\"}",
                "{\"thought\":\"Let me run VectorDBTool to get more information\",\"action\":\"vectordbtool\"}"
            );
        Assert.assertTrue(expected.contains(output.get(THOUGHT_RESPONSE)));
    }

    @Test
    public void testParseLLMOutput_MultipleFields_NoActionAndFinalAnswer() {
        Set<String> tools = Set.of("VectorDBTool", "ListIndexTool");
        String key1 = "dummy key1";
        String value1 = "dummy value1";
        String key2 = "dummy key2";
        String value2 = "dummy value2";
        ModelTensorOutput modelTensoOutput = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(
                                List.of(ModelTensor.builder().name("response").dataAsMap(Map.of(key1, value1, key2, value2)).build())
                            )
                            .build()
                    )
            )
            .build();
        Map<String, String> output = AgentUtils
            .parseLLMOutput(Collections.emptyMap(), modelTensoOutput, null, tools, Collections.emptyList());
        Assert.assertEquals(2, output.size());
        Assert.assertFalse(output.containsKey(THOUGHT));
        Assert.assertFalse(output.containsKey(ACTION));
        Set<String> expected = Set
            .of(
                "{\"dummy key1\":\"dummy value1\",\"dummy key2\":\"dummy value2\"}",
                "{\"dummy key2\":\"dummy value2\",\"dummy key1\":\"dummy value1\"}"
            );
        Assert.assertTrue(expected.contains(output.get(THOUGHT_RESPONSE)));
        Assert.assertEquals(output.get(THOUGHT_RESPONSE), output.get(FINAL_ANSWER));
    }

    @Test
    public void testParseLLMOutput_OneFields_NoActionAndFinalAnswer() {
        Set<String> tools = Set.of("VectorDBTool", "ListIndexTool");
        String thought = "Let me run VectorDBTool to get more information";
        ModelTensorOutput modelTensoOutput = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(Map.of(THOUGHT, thought)).build()))
                            .build()
                    )
            )
            .build();
        Map<String, String> output = AgentUtils
            .parseLLMOutput(Collections.emptyMap(), modelTensoOutput, null, tools, Collections.emptyList());
        Assert.assertEquals(3, output.size());
        Assert.assertEquals(thought, output.get(THOUGHT));
        Assert.assertFalse(output.containsKey(ACTION));
        Assert.assertEquals("{\"thought\":\"Let me run VectorDBTool to get more information\"}", output.get(THOUGHT_RESPONSE));
        Assert.assertEquals("{\"thought\":\"Let me run VectorDBTool to get more information\"}", output.get(FINAL_ANSWER));
    }

    @Test
    public void testExtractThought_InvalidResult() {
        String text = responseForActionInvalidJson;
        String result = AgentUtils.extractThought(text);
        Assert.assertEquals("Let me search our index to find population projections", result);
    }

    @Test
    public void testConstructToolParams() {
        String question = "dummy question";
        String actionInput = "{'detectorName': 'abc', 'indices': 'sample-data' }";
        verifyConstructToolParams(question, actionInput, (toolParams) -> {
            Assert.assertEquals(5, toolParams.size());
            Assert.assertEquals(actionInput, toolParams.get("input"));
            Assert.assertEquals("abc", toolParams.get("detectorName"));
            Assert.assertEquals("sample-data", toolParams.get("indices"));
            Assert.assertEquals("value1", toolParams.get("key1"));
            Assert.assertEquals(actionInput, toolParams.get(LLM_GEN_INPUT));
        });
    }

    @Test
    public void testConstructToolParamsNullActionInput() {
        String question = "dummy question";
        String actionInput = null;
        verifyConstructToolParams(question, actionInput, (toolParams) -> {
            Assert.assertEquals(3, toolParams.size());
            Assert.assertEquals("value1", toolParams.get("key1"));
            Assert.assertNull(toolParams.get(LLM_GEN_INPUT));
            Assert.assertNull(toolParams.get("input"));
        });
    }

    @Test
    public void testConstructToolParams_UseOriginalInput() {
        String question = "dummy question";
        String actionInput = "{'detectorName': 'abc', 'indices': 'sample-data' }";
        when(tool1.useOriginalInput()).thenReturn(true);
        verifyConstructToolParams(question, actionInput, (toolParams) -> {
            Assert.assertEquals(5, toolParams.size());
            Assert.assertEquals(question, toolParams.get("input"));
            Assert.assertEquals("value1", toolParams.get("key1"));
            Assert.assertEquals(actionInput, toolParams.get(LLM_GEN_INPUT));
            Assert.assertEquals("sample-data", toolParams.get("indices"));
            Assert.assertEquals("abc", toolParams.get("detectorName"));
        });
    }

    @Test
    public void testConstructToolParams_PlaceholderConfigInput() {
        String question = "dummy question";
        String actionInput = "action input";
        String preConfigInputStr = "Config Input: ";
        Map<String, Tool> tools = Map.of("tool1", tool1);
        Map<String, MLToolSpec> toolSpecMap = Map
            .of(
                "tool1",
                MLToolSpec
                    .builder()
                    .type("tool1")
                    .parameters(Map.of("key1", "value1"))
                    .configMap(Map.of("input", preConfigInputStr + "${parameters.llm_generated_input}"))
                    .build()
            );
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        String action = "tool1";
        Map<String, String> toolParams = AgentUtils.constructToolParams(tools, toolSpecMap, question, lastActionInput, action, actionInput);
        Assert.assertEquals(3, toolParams.size());
        Assert.assertEquals(preConfigInputStr + actionInput, toolParams.get("input"));
        Assert.assertEquals("value1", toolParams.get("key1"));
        Assert.assertEquals(actionInput, toolParams.get(LLM_GEN_INPUT));
    }

    @Test
    public void testConstructToolParams_PlaceholderConfigInputJson() {
        String question = "dummy question";
        String actionInput = "{'detectorName': 'abc', 'indices': 'sample-data' }";
        String preConfigInputStr = "Config Input: ";
        Map<String, Tool> tools = Map.of("tool1", tool1);
        Map<String, MLToolSpec> toolSpecMap = Map
            .of(
                "tool1",
                MLToolSpec
                    .builder()
                    .type("tool1")
                    .parameters(Map.of("key1", "value1"))
                    .configMap(Map.of("input", preConfigInputStr + "${parameters.detectorName}"))
                    .build()
            );
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        String action = "tool1";
        Map<String, String> toolParams = AgentUtils.constructToolParams(tools, toolSpecMap, question, lastActionInput, action, actionInput);
        Assert.assertEquals(5, toolParams.size());
        Assert.assertEquals(preConfigInputStr + "abc", toolParams.get("input"));
        Assert.assertEquals("value1", toolParams.get("key1"));
        Assert.assertEquals(actionInput, toolParams.get(LLM_GEN_INPUT));
    }

    @Test
    public void testParseLLMOutputWithOpenAIFormat() {
        Map<String, String> parameters = Map
            .of(
                LLM_RESPONSE_FILTER,
                "$.choices[0].message.content",
                TOOL_CALLS_PATH,
                "$.choices[0].message.tool_calls",
                TOOL_CALLS_TOOL_NAME,
                "function.name",
                TOOL_CALLS_TOOL_INPUT,
                "function.arguments",
                TOOL_CALL_ID_PATH,
                "id",
                LLM_FINISH_REASON_PATH,
                "$.choices[0].finish_reason",
                LLM_FINISH_REASON_TOOL_USE,
                "tool_calls"
            );

        // Test case 1: Response containing both text and toolUse
        Map<String, Object> responseData1 = Map
            .of(
                "choices",
                List
                    .of(
                        Map
                            .of(
                                "message",
                                Map
                                    .of(
                                        "content",
                                        "I will use ListIndexTool",
                                        "tool_calls",
                                        List
                                            .of(
                                                Map
                                                    .of(
                                                        "function",
                                                        Map.of("name", "ListIndexTool", "arguments", "{\"indices\":[]}"),
                                                        "id",
                                                        "tool_1"
                                                    )
                                            )
                                    ),
                                "finish_reason",
                                "tool_calls"
                            )
                    )
            );

        ModelTensorOutput modelTensorOutput1 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData1).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output1 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput1, null, Set.of("ListIndexTool"), new ArrayList<>());

        Assert.assertEquals("", output1.get(THOUGHT));
        Assert.assertEquals("ListIndexTool", output1.get(ACTION));
        Assert.assertEquals("{\"indices\":[]}", output1.get(ACTION_INPUT));
        Assert.assertEquals("tool_1", output1.get(TOOL_CALL_ID));

        // Test case 2: Response containing only toolUse
        Map<String, Object> responseData2 = Map
            .of(
                "choices",
                List
                    .of(
                        Map
                            .of(
                                "message",
                                Map
                                    .of(
                                        "tool_calls",
                                        List
                                            .of(
                                                Map
                                                    .of(
                                                        "function",
                                                        Map.of("name", "IndexMappingTool", "arguments", "{\"index\":[\"test_index\"]}"),
                                                        "id",
                                                        "tool_2"
                                                    )
                                            )
                                    ),
                                "finish_reason",
                                "tool_calls"
                            )
                    )
            );

        ModelTensorOutput modelTensorOutput2 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData2).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output2 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput2, null, Set.of("IndexMappingTool"), new ArrayList<>());

        Assert.assertEquals("", output2.get(THOUGHT));
        Assert.assertEquals("IndexMappingTool", output2.get(ACTION));
        Assert.assertEquals("{\"index\":[\"test_index\"]}", output2.get(ACTION_INPUT));
        Assert.assertEquals("tool_2", output2.get(TOOL_CALL_ID));

        // Test case 3: Response containing only text
        Map<String, Object> responseData3 = Map
            .of("choices", List.of(Map.of("message", Map.of("content", "This is a test response"), "finish_reason", "stop")));

        ModelTensorOutput modelTensorOutput3 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData3).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output3 = AgentUtils.parseLLMOutput(parameters, modelTensorOutput3, null, Set.of(), new ArrayList<>());

        Assert.assertNull(output3.get(ACTION));
        Assert.assertNull(output3.get(ACTION_INPUT));
        Assert.assertNull(output3.get(TOOL_CALL_ID));
        Assert.assertTrue(output3.get(FINAL_ANSWER).contains("This is a test response"));
    }

    @Test
    public void testParseLLMOutputWithClaudeFormat() {
        Map<String, String> parameters = Map
            .of(
                LLM_RESPONSE_FILTER,
                "$.output.message.content[0].text",
                TOOL_CALLS_PATH,
                "$.output.message.content[*].toolUse",
                TOOL_CALLS_TOOL_NAME,
                "name",
                TOOL_CALLS_TOOL_INPUT,
                "input",
                TOOL_CALL_ID_PATH,
                "toolUseId",
                LLM_FINISH_REASON_PATH,
                "$.stopReason",
                LLM_FINISH_REASON_TOOL_USE,
                "tool_use"
            );

        // Test case 1: Response containing both text and toolUse
        Map<String, Object> responseData1 = Map
            .of(
                "output",
                Map
                    .of(
                        "message",
                        Map
                            .of(
                                "content",
                                List
                                    .of(
                                        Map.of("text", "I will use ListIndexTool"),
                                        Map
                                            .of(
                                                "toolUse",
                                                Map
                                                    .of(
                                                        "input",
                                                        Map.of("indices", List.of()),
                                                        "name",
                                                        "ListIndexTool",
                                                        "toolUseId",
                                                        "tool_1"
                                                    )
                                            )
                                    )
                            )
                    ),
                "stopReason",
                "tool_use"
            );

        ModelTensorOutput modelTensorOutput1 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData1).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output1 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput1, null, Set.of("ListIndexTool"), new ArrayList<>());

        Assert.assertEquals("", output1.get(THOUGHT));
        Assert.assertEquals("ListIndexTool", output1.get(ACTION));
        Assert.assertEquals("{\"indices\":[]}", output1.get(ACTION_INPUT));
        Assert.assertEquals("tool_1", output1.get(TOOL_CALL_ID));

        // Test case 2: Response containing only toolUse
        Map<String, Object> responseData2 = Map
            .of(
                "output",
                Map
                    .of(
                        "message",
                        Map
                            .of(
                                "content",
                                List
                                    .of(
                                        Map
                                            .of(
                                                "toolUse",
                                                Map
                                                    .of(
                                                        "input",
                                                        Map.of("index", List.of("test_index")),
                                                        "name",
                                                        "IndexMappingTool",
                                                        "toolUseId",
                                                        "tool_2"
                                                    )
                                            )
                                    )
                            )
                    ),
                "stopReason",
                "tool_use"
            );

        ModelTensorOutput modelTensorOutput2 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData2).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output2 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput2, null, Set.of("IndexMappingTool"), new ArrayList<>());

        Assert.assertEquals("", output2.get(THOUGHT));
        Assert.assertEquals("IndexMappingTool", output2.get(ACTION));
        Assert.assertEquals("{\"index\":[\"test_index\"]}", output2.get(ACTION_INPUT));
        Assert.assertEquals("tool_2", output2.get(TOOL_CALL_ID));

        // Test case 3: Response containing only text
        Map<String, Object> responseData3 = Map
            .of("output", Map.of("message", Map.of("content", List.of(Map.of("text", "This is a test response")))), "stopReason", "stop");

        ModelTensorOutput modelTensorOutput3 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData3).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output3 = AgentUtils.parseLLMOutput(parameters, modelTensorOutput3, null, Set.of(), new ArrayList<>());

        Assert.assertNull(output3.get(ACTION));
        Assert.assertNull(output3.get(ACTION_INPUT));
        Assert.assertNull(output3.get(TOOL_CALL_ID));
        Assert.assertTrue(output3.get(FINAL_ANSWER).contains("This is a test response"));
    }

    @Test
    public void testParseLLMOutputWithDeepseekFormat() {
        Map<String, String> parameters = Map
            .of(
                LLM_RESPONSE_FILTER,
                "$.output.message.content[0].text",
                TOOL_CALLS_PATH,
                "_llm_response.tool_calls",
                TOOL_CALLS_TOOL_NAME,
                "tool_name",
                TOOL_CALLS_TOOL_INPUT,
                "input",
                TOOL_CALL_ID_PATH,
                "id",
                LLM_FINISH_REASON_PATH,
                "_llm_response.stop_reason",
                LLM_FINISH_REASON_TOOL_USE,
                "tool_use"
            );

        // Test case 1: Response containing both text and tool use
        Map<String, Object> responseData1 = Map
            .of(
                "output",
                Map
                    .of(
                        "message",
                        Map
                            .of(
                                "content",
                                List
                                    .of(
                                        Map
                                            .of(
                                                "text",
                                                "{\"stop_reason\": \"tool_use\", \"tool_calls\": [{\"id\":\"tool_1\",\"tool_name\":\"ListIndexTool\",\"input\": {\"indices\":[]}}]}"
                                            )
                                    )
                            )
                    )
            );

        ModelTensorOutput modelTensorOutput1 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData1).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output1 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput1, null, Set.of("ListIndexTool"), new ArrayList<>());

        Assert.assertEquals("", output1.get(THOUGHT));
        Assert.assertEquals("ListIndexTool", output1.get(ACTION));
        Assert.assertEquals("{\"indices\":[]}", output1.get(ACTION_INPUT));
        Assert.assertEquals("tool_1", output1.get(TOOL_CALL_ID));

        // Test case 2: Response containing only tool use
        Map<String, Object> responseData2 = Map
            .of(
                "output",
                Map
                    .of(
                        "message",
                        Map
                            .of(
                                "content",
                                List
                                    .of(
                                        Map
                                            .of(
                                                "text",
                                                "{\"stop_reason\": \"tool_use\", \"tool_calls\": [{\"id\":\"tool_2\",\"tool_name\":\"IndexMappingTool\",\"input\": {\"index\":[\"test_index\"]}}]}"
                                            )
                                    )
                            )
                    )
            );

        ModelTensorOutput modelTensorOutput2 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData2).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output2 = AgentUtils
            .parseLLMOutput(parameters, modelTensorOutput2, null, Set.of("IndexMappingTool"), new ArrayList<>());

        Assert.assertEquals("", output2.get(THOUGHT));
        Assert.assertEquals("IndexMappingTool", output2.get(ACTION));
        Assert.assertEquals("{\"index\":[\"test_index\"]}", output2.get(ACTION_INPUT));
        Assert.assertEquals("tool_2", output2.get(TOOL_CALL_ID));

        // Test case 3: Response containing only text (final answer)
        Map<String, Object> responseData3 = Map
            .of(
                "output",
                Map
                    .of(
                        "message",
                        Map
                            .of(
                                "content",
                                List
                                    .of(
                                        Map
                                            .of(
                                                "text",
                                                "{\"stop_reason\": \"end_turn\", \"message\": {\"content\":[{\"text\":\"This is a test response\"}]}}"
                                            )
                                    )
                            )
                    )
            );

        ModelTensorOutput modelTensorOutput3 = ModelTensorOutput
            .builder()
            .mlModelOutputs(
                List
                    .of(
                        ModelTensors
                            .builder()
                            .mlModelTensors(List.of(ModelTensor.builder().name("response").dataAsMap(responseData3).build()))
                            .build()
                    )
            )
            .build();

        Map<String, String> output3 = AgentUtils.parseLLMOutput(parameters, modelTensorOutput3, null, Set.of(), new ArrayList<>());

        Assert.assertNull(output3.get(ACTION));
        Assert.assertNull(output3.get(ACTION_INPUT));
        Assert.assertNull(output3.get(TOOL_CALL_ID));
        Assert.assertTrue(output3.get(FINAL_ANSWER).contains("This is a test response"));
    }

    private void verifyConstructToolParams(String question, String actionInput, Consumer<Map<String, String>> verify) {
        Map<String, Tool> tools = Map.of("tool1", tool1);
        Map<String, MLToolSpec> toolSpecMap = Map
            .of("tool1", MLToolSpec.builder().type("tool1").parameters(Map.of("key1", "value1")).build());
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        String action = "tool1";
        Map<String, String> toolParams = AgentUtils.constructToolParams(tools, toolSpecMap, question, lastActionInput, action, actionInput);
        verify.accept(toolParams);
    }
}
