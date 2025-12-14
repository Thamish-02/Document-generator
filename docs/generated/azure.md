## AI Summary

A file named azure.py.


## Class: MutuallyExclusiveAuthError

## Class: BaseAzureClient

## Class: AzureOpenAI

## Class: AsyncAzureOpenAI

### Function: __init__(self)

### Function: _build_request(self, options)

### Function: _prepare_url(self, url)

**Description:** Adjust the URL if the client was configured with an Azure endpoint + deployment
and the API feature being called is **not** a deployments-based endpoint
(i.e. requires /deployments/deployment-name in the URL path).

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

**Description:** Construct a new synchronous azure openai client instance.

This automatically infers the following arguments from their corresponding environment variables if they are not provided:
- `api_key` from `AZURE_OPENAI_API_KEY`
- `organization` from `OPENAI_ORG_ID`
- `project` from `OPENAI_PROJECT_ID`
- `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
- `api_version` from `OPENAI_API_VERSION`
- `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`

Args:
    azure_endpoint: Your Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`

    azure_ad_token: Your Azure Active Directory token, https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id

    azure_ad_token_provider: A function that returns an Azure Active Directory token, will be invoked on every request.

    azure_deployment: A model deployment, if given with `azure_endpoint`, sets the base client URL to include `/deployments/{azure_deployment}`.
        Not supported with Assistants APIs.

### Function: copy(self)

**Description:** Create a new client instance re-using the same options given to the current client with optional overriding.

### Function: _get_azure_ad_token(self)

### Function: _prepare_options(self, options)

### Function: _configure_realtime(self, model, extra_query)

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

**Description:** Construct a new asynchronous azure openai client instance.

This automatically infers the following arguments from their corresponding environment variables if they are not provided:
- `api_key` from `AZURE_OPENAI_API_KEY`
- `organization` from `OPENAI_ORG_ID`
- `project` from `OPENAI_PROJECT_ID`
- `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
- `api_version` from `OPENAI_API_VERSION`
- `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`

Args:
    azure_endpoint: Your Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`

    azure_ad_token: Your Azure Active Directory token, https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id

    azure_ad_token_provider: A function that returns an Azure Active Directory token, will be invoked on every request.

    azure_deployment: A model deployment, if given with `azure_endpoint`, sets the base client URL to include `/deployments/{azure_deployment}`.
        Not supported with Assistants APIs.

### Function: copy(self)

**Description:** Create a new client instance re-using the same options given to the current client with optional overriding.
