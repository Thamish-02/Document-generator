## AI Summary

A file named _state.py.


## Class: CLIENT

## Class: SERVER

## Class: IDLE

## Class: SEND_RESPONSE

## Class: SEND_BODY

## Class: DONE

## Class: MUST_CLOSE

## Class: CLOSED

## Class: ERROR

## Class: MIGHT_SWITCH_PROTOCOL

## Class: SWITCHED_PROTOCOL

## Class: _SWITCH_UPGRADE

## Class: _SWITCH_CONNECT

## Class: ConnectionState

### Function: __init__(self)

### Function: process_error(self, role)

### Function: process_keep_alive_disabled(self)

### Function: process_client_switch_proposal(self, switch_event)

### Function: process_event(self, role, event_type, server_switch_event)

### Function: _fire_event_triggered_transitions(self, role, event_type)

### Function: _fire_state_triggered_transitions(self)

### Function: start_next_cycle(self)
