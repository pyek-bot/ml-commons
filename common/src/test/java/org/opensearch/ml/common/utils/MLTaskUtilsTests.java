/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.utils;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.opensearch.common.action.ActionFuture;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.transport.task.MLTaskGetAction;
import org.opensearch.ml.common.transport.task.MLTaskGetRequest;
import org.opensearch.ml.common.transport.task.MLTaskGetResponse;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

@RunWith(MockitoJUnitRunner.class)
public class MLTaskUtilsTests {
    private Client client;
    private ThreadPool threadPool;
    private ThreadContext threadContext;

    @Before
    public void setup() {
        this.client = mock(Client.class);
        this.threadPool = mock(ThreadPool.class);
        Settings settings = Settings.builder().build();
        this.threadContext = new ThreadContext(settings);
        when(client.threadPool()).thenReturn(threadPool);
        when(threadPool.getThreadContext()).thenReturn(threadContext);
    }

    @Test
    public void testIsTaskMarkedForCancel() {
        String taskId = "testTaskId";
        MLTask mlTask = mock(MLTask.class);
        when(mlTask.getState()).thenReturn(MLTaskState.CANCELLING);
        MLTaskGetResponse mlTaskGetResponse = mock(MLTaskGetResponse.class);
        when(mlTaskGetResponse.getMlTask()).thenReturn(mlTask);

        ActionFuture<MLTaskGetResponse> actionFuture = mock(ActionFuture.class);
        when(actionFuture.actionGet()).thenReturn(mlTaskGetResponse);
        when(client.execute(any(MLTaskGetAction.class), any(MLTaskGetRequest.class))).thenReturn(actionFuture);

        assertTrue(MLTaskUtils.isTaskMarkedForCancel(taskId, client));
    }

    @Test
    public void testIsTaskNotMarkedForCancel() {
        String taskId = "testTaskId";
        MLTask mlTask = mock(MLTask.class);
        when(mlTask.getState()).thenReturn(MLTaskState.RUNNING);
        MLTaskGetResponse mlTaskGetResponse = mock(MLTaskGetResponse.class);
        when(mlTaskGetResponse.getMlTask()).thenReturn(mlTask);

        ActionFuture<MLTaskGetResponse> actionFuture = mock(ActionFuture.class);
        when(actionFuture.actionGet()).thenReturn(mlTaskGetResponse);
        when(client.execute(any(MLTaskGetAction.class), any(MLTaskGetRequest.class))).thenReturn(actionFuture);

        assertFalse(MLTaskUtils.isTaskMarkedForCancel(taskId, client));
    }
}
