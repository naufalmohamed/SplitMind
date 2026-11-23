from splitmind.messages import KEY_TOPIC, Message


def msg(source, value, created_at, confidence, sent_at=None, version=0, mid=None, relevance=1.0):
    return Message(
        source=source,
        content={KEY_TOPIC: value},
        created_at=created_at,
        sent_at=created_at if sent_at is None else sent_at,
        confidence=confidence,
        world_version=version,
        message_id=mid or f"{source}:{created_at}:{value}",
        relevance=relevance,
    )
