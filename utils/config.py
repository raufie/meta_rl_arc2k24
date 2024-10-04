import yaml
class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(data)