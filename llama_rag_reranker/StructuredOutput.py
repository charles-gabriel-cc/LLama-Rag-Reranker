import instructor
from pydantic import BaseModel, Field
from openai import OpenAI

class output_format(BaseModel):
    isRelevant: bool = Field(False, description="Variable that determines whether the information content is relevant to answer the user's query, True for yes and False for no")

class StructuredOutput():
    def __init__(self, output_format: BaseModel = output_format, **kwargs) -> None:
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
        self.output_format = output_format

    def query_output(self, output: str):
        response = self.client.chat.completions.create(
            #setar atributo context
            model="llama3.2:3b-instruct-fp16",
            messages=[
                {
                    "role": "user",
                    "content": output,
                }
            ],
            response_model=self.output_format,
        )
        return response
    
if __name__ == '__main__':

    aux = StructuredOutput()

    response = aux.query_output("Relevant information")
    print(response)
    
