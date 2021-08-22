from typing import List, Any


class Merge:
    def __init__(self, first_element, second_element, first_idx=None, second_idx=None):
        self.first_element = first_element
        self.second_element = second_element
        self.first_idx = first_idx
        self.second_idx = second_idx
        self.prev = None
        self.next = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])

        first = self.first_element([])
        second = self.second_element([])
        if self.first_idx is not None:
            first = [first[self.first_idx]]

        if self.second_idx is not None:
            second = [second[self.second_idx]]

        captures.extend(first)
        captures.extend(second)

        return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
