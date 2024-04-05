import typer
import time
from typing import Optional
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List, ClassVar, Optional
from datasynth.base import BaseChain
from typing import ClassVar
from datasynth.few_shot import generate_population, populate_few_shot


class DatasetPipeline(BaseChain):

    k: int = 10
    sample_size: int = 3
    dataset_name: Optional[str] = None
    manual_review: bool = False
    generator: ClassVar[GeneratorChain]
    normalizer: ClassVar[NormalizerChain]
    chain_type = "DatasetPipeline"

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return ["dataset"]

    def run(
        self, inputs: dict[str, str] | None = None
    ) -> dict[str, List[dict[str, Any | str]]]:
        """
        Executes the dataset generation pipeline. This method is the primary entry point for running
        the dataset generation and normalization process. It delegates to the _call method with the
        provided inputs, ensuring that inputs are always in the expected format for the generator and
        normalizer chains.

        Args:
            inputs (Optional[dict[str, str]]): Optional dictionary of inputs to pass to the generator. Defaults to None.

        Returns:
            dict[str, List[dict[str, Any | str]]]: The results of the dataset generation and normalization process.
        """

        return self._call(inputs)

    def _call(
        self,
        inputs: Optional[dict[str, str] | None] = None,
    ) -> dict[str, List[dict[str, Any | str]]]:
        generated: List[dict[str, Any | str]] = []
        population = generate_population(self.datatype)
        while len(generated) < self.k:
            few_shot = populate_few_shot(
                population=population, sample_size=self.sample_size
            )
            inputs = inputs or {}
            inputs["few_shot"] = few_shot

            generated_content: dict[str, str] = self.generator.invoke(input=inputs)
            if generated_content.get("generated") is not None:
                print("generated content:", generated_content)
                generated.extend(generated_content["generated"])
            else:
                print("warning: No content generated")
                break
        outputs = []
        for example in generated[: self.k]:
            try:
                normalizer_inputs: dict[str, Any] = {
                    self.normalizer.input_keys[0]: example.strip()
                }
                print("EXAMPLE INPUTS", example)
                print("NORMALIZER INPUTS", normalizer_inputs)
                outputs.append(
                    {
                        "generated_input": example,
                        "normalized_output": self.normalizer.invoke(normalizer_inputs),
                    }
                )
            except:
                continue
        results: dict[str, dict[str, list[dict[str, Any | str]] | str]] = {
            "dataset": {
                "outputs": outputs,
                "generator_prompt": self.generator.template.template,
                "normalizer_prompt": self.normalizer.template.template,
            }
        }

        assert len(results["dataset"]["outputs"]) == self.k

        if self.manual_review:
            for pair in results["dataset"]["outputs"]:

                print(
                    f"{'Generated Input:', pair['generated_input']}\n-------\n{'Normalized Output:', pair['normalized_output']['normalized']}\n--------"
                )
                user_input = input("Accept? (y/N): ")
                if user_input.lower() == "y":
                    pair["manual_review"] = "accepted"
                else:
                    pair["manual_review"] = "rejected"

        if self.dataset_name is None:
            self.dataset_name = f"{str(int(time.time()))}.json"
        else:
            self.dataset_name = f"{self.dataset_name}.json"

        dataset_dir: str = os.path.join(os.path.dirname(__file__), "datasets")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        dataset_path: str = f"{dataset_dir}/{self.dataset_name}"
        with open(dataset_path, "w") as fd:
            json.dump(results, fd)

        return results


# TODO: look at overlap of generator and normalizer dirs
template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(
    template_dir,
    DatasetPipeline,
    "DatasetPipeline",
    generator_chain=GeneratorChain,
    normalizer_chain=NormalizerChain,
)


def generate_dataset(
    datatype: str,
    k: int = 10,
    dataset_name: str = None,
    temperature: float = 0.8,
    cache: bool = False,
    manual_review: bool = False,
):
    """Generate synthetic data and the normalized output

    Args:
        datatype (str): Name of the data to be generated, given by the template's name.
        k (int, optional): Number of samples to generate | Defaults to 10.
        dataset_name (str, optional): Name of generated data (user defined) | Defaults to None.
        temperature (float, optional): Parameter that affects the randomness of the output | Defaults to 0.8.
        cache (bool, optional): Determine wheher to cache calls to/from the API | Defaults to False.

    Returns:
        dict[str, dict[str, list[dict[str, Any | str]] | str]] : A dictionary of synthetically generated and normalized data
    """

    auto_class(
        template_dir,
        DatasetPipeline,
        "DatasetPipeline",
        generator_chain=GeneratorChain,
        normalizer_chain=NormalizerChain,
        temperature=temperature,
        cache=cache,
    )
    chain = DatasetPipeline.from_name(
        datatype, k=k, dataset_name=dataset_name, manual_review=manual_review
    )
    
    return chain.run()


if __name__ == "__main__":
    typer.run(generate_dataset)
