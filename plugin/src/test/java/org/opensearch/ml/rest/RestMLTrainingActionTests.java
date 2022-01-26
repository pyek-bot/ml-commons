/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.rest;

import java.util.List;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.common.Strings;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.test.OpenSearchTestCase;

public class RestMLTrainingActionTests extends OpenSearchTestCase {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private RestMLTrainingAction restMLTrainingAction;

    @Before
    public void setup() {
        restMLTrainingAction = new RestMLTrainingAction();
    }

    @Test
    public void testConstructor() {
        RestMLTrainingAction mlTrainingAction = new RestMLTrainingAction();
        assertNotNull(mlTrainingAction);
    }

    @Test
    public void testGetName() {
        String actionName = restMLTrainingAction.getName();
        assertFalse(Strings.isNullOrEmpty(actionName));
        assertEquals("ml_training_action", actionName);
    }

    @Test
    public void testRoutes() {
        List<RestHandler.Route> routes = restMLTrainingAction.routes();
        assertNotNull(routes);
        assertFalse(routes.isEmpty());
        RestHandler.Route route = routes.get(0);
        assertEquals(RestRequest.Method.POST, route.getMethod());
        assertEquals("/_plugins/_ml/_train/{algorithm}", route.getPath());
    }
}