import math
import re
from datetime import datetime

def extract_time(time_string: str) -> int:
    """Convert time_strings to integer timestamps
    Input:
        time_string: a string like '[2023-10-13 21:16:08.727]'.
    Return:
        An integer representing number of seconds elapsed since 1970/1/1.
    >>> extract_time("[1970-01-03 00:00:00.000]")
    144000
    >>> extract_time("[2023-10-13 21:16:08.727]")
    1697202969
    """
    time_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]')
    time_match = time_pattern.search(time_string)
    if time_match:
        time_str = time_match.group(1)
        time_extracted = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        return math.ceil(time_extracted.timestamp())
    else:
        raise ValueError("Not a string of time.")

if __name__ == "__main__":
    """"""
    import doctest
    doctest.testmod()