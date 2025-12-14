## AI Summary

A file named _exceptions.py.


## Class: OpenAIError

## Class: APIError

## Class: APIResponseValidationError

## Class: APIStatusError

**Description:** Raised when an API response has a status code of 4xx or 5xx.

## Class: APIConnectionError

## Class: APITimeoutError

## Class: BadRequestError

## Class: AuthenticationError

## Class: PermissionDeniedError

## Class: NotFoundError

## Class: ConflictError

## Class: UnprocessableEntityError

## Class: RateLimitError

## Class: InternalServerError

## Class: LengthFinishReasonError

## Class: ContentFilterFinishReasonError

## Class: InvalidWebhookSignatureError

**Description:** Raised when a webhook signature is invalid, meaning the computed signature does not match the expected signature.

### Function: __init__(self, message, request)

### Function: __init__(self, response, body)

### Function: __init__(self, message)

### Function: __init__(self)

### Function: __init__(self, request)

### Function: __init__(self)

### Function: __init__(self)
