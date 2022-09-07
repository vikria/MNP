from abc import abstractmethod, ABC
import re
import os.path

"""Classes to have unified interface of data access."""


class DataSource(ABC):
    """ Base class for all data sources. """

    def __init__(self, item_type: type = int):
        self.item_type = item_type

    @abstractmethod
    def get_data(self) -> list:
        raise NotImplementedError()


class ExternalSource(DataSource):
    """ Class to work with external (in-python-already) data. """

    def __init__(self, data: list = None, **kwargs):
        if not data:
            raise ValueError(f'Data must be set.')

        super(ExternalSource, self).__init__(**kwargs)
        self.data = data

    def get_data(self) -> list:
        return self.data


class FileSource(DataSource):
    """ Class to work with file sources. """

    def __init__(self, filepath: str = None, **kwargs):
        if not filepath:
            raise ValueError(f'File path must be set.')
        if not os.path.isfile(filepath):
            raise ValueError(f'There is no such file {filepath}.')

        super(FileSource, self).__init__(**kwargs)
        self.filepath = filepath

    def get_data(self) -> list:
        with open(self.filepath, 'r') as file:
            _data = re.findall(r'[0-9]+', file.read())

        return list(map(self.item_type, _data))
