from paomcp.tools.backends.gdb import GdbController


def bare_controller():
    controller = object.__new__(GdbController)
    controller._state = "running"
    controller._inferior_pid = None
    controller._inferior_epoch = 0
    return controller


def test_stopped_notification_allows_null_payload():
    controller = bare_controller()

    controller._handle_notify({"type": "notify", "message": "stopped", "payload": None})

    assert controller.get_state() == "stopped"


def test_thread_group_started_allows_null_payload():
    controller = bare_controller()

    controller._handle_notify(
        {"type": "notify", "message": "thread-group-started", "payload": None}
    )

    assert controller.get_inferior_pid() is None
    assert controller.get_inferior_epoch() == 1


def test_thread_group_exit_advances_inferior_epoch():
    controller = bare_controller()

    controller._handle_notify(
        {"type": "notify", "message": "thread-group-exited", "payload": {}}
    )

    assert controller.get_state() == "exited"
    assert controller.get_inferior_epoch() == 1
