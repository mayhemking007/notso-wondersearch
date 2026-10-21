# Prevent duplicate orders

An idempotency key identifies one intended operation. Store the key with the
order and enforce a unique database constraint. If the client retries the same
request, return the stored response instead of creating another order.
Request deduplication makes retries safe after a network timeout.
