import os
from typing import ClassVar
from pydantic import Field, PrivateAttr
from langchain.chains import LLMChain
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
import typer
from datasynth import TEMPLATE_DIR
from pprint import pprint
from datasynth.base import BaseChain, auto_class
from few_shot import populate_few_shot, generate_population
from settings import SEPARATOR

separator = SEPARATOR


class GeneratorChain(BaseChain):
    template: PromptTemplate = PrivateAttr()
    chain = Field(LLMChain, required=False)
    chain_type: ClassVar[str] = "generator"
    temperature: float = 0.0
    cache: bool = False
    verbose: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = PromptTemplate(
            input_variables=["few_shot"],
            template=open(
                os.path.join(TEMPLATE_DIR, "generator", f"{self.datatype}.template")
            ).read(),
            validate_template=False,
            template_format="jinja2",
        )
        if self.temperature > 2.0:
            raise ValueError(
                f"temperature:{self.temperature} is greater than the maximum of 2-'temperature'"
            )
        self.chain = LLMChain(
            prompt=self.template,
            llm=OpenAI(temperature=self.temperature, cache=self.cache),
            verbose=self.verbose,
        )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["generated"]
    
    def _call(self, inputs: dict[str, str]) -> dict[str, list[str]]:
        print(f"inputs: {inputs}")
        generated_items = self.chain.invoke(input=inputs)["text"].split(separator)
        print(f"generated_items: {generated_items}")
        filtered_items = [item for item in generated_items if item.strip()]
        return {"generated": filtered_items}


template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(template_dir, GeneratorChain, "Generator")


def main(
    datatype: str,
    sample_size: int = 3,
    temperature: float = 0.5,
    cache: bool = False,
    verbose: bool = True,
):
    population = generate_population(datatype)
    few_shot = populate_few_shot(population=population, sample_size=sample_size)
    chain = GeneratorChain.from_name(
        datatype, temperature=temperature, cache=cache, verbose=verbose
    )
    pprint(chain.invoke(few_shot))


if __name__ == "__main__":
    typer.run(main)
