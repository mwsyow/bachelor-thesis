import json

class Config:
    def __init__(self, filename="configs"):
        self.filename = filename
        self.config = self._read(filename)

    def __str__(self):
        return f"Filename: {self.filename}, Dict: {self.config}"

    def _read(self, fn):    
        # Opening JSON file
        with open(fn, 'r') as file:
            data = json.load(file)
            file.close()
            return data

if __name__ == "__main__":
    print(Config())
