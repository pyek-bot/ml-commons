package org.opensearch.ml.common.agent.langgraph;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.data.message.UserMessage;
import org.bsc.langgraph4j.CompileConfig;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.RunnableConfig;
import org.bsc.langgraph4j.agentexecutor.AgentExecutor;
import org.bsc.langgraph4j.checkpoint.MemorySaver;

import java.util.Map;

public class TestAgent {

    public static void main(String[] args) throws GraphStateException {
        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_KEY"))
                .modelName( "gpt-4o-mini" )
                .logResponses(true)
                .maxRetries(2)
                .temperature(0.0)
                .maxTokens(2000)
                .build();

        var stateGraph = AgentExecutor.builder()
                .chatModel(model)
                .toolsFromObject( new TestTool() )
                .build();

        var saver = new MemorySaver();

        CompileConfig compileConfig = CompileConfig.builder()
                .checkpointSaver( saver )
                .build();

        var graph = stateGraph.compile( compileConfig );

        var config = RunnableConfig.builder()
                .threadId("test1")
                .build();

        var iterator = graph.streamSnapshots( Map.of( "messages", UserMessage.from("perform test once")), config );

        for( var step : iterator ) {
            System.out.printf("STEP: %s", step);
        }
    }
}
