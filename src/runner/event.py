class Event(object):
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __str__(self):
        return "Event: {0} {1}".format(self.name, self.data)


def first_win(episodes, number_of_steps, cumulative):
    return Event(
        "first_win", {"episodes": episodes, "number_of_steps": number_of_steps, "cumulative": list(cumulative.values())[0]}
    )


def win(episodes, number_of_steps, cumulative):
    return Event("win", {"episodes": episodes, "number_of_steps": number_of_steps, "cumulative": list(cumulative.values())[0]})
