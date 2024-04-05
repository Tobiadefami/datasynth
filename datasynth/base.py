from typing import ClassVar
from langchain.chains.base import Chain
from typing import Any
import os
import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache()


class BaseChain(Chain):
    datatype: ClassVar[str]
    chain_type: ClassVar[str]
    registry: ClassVar[dict[Any, str]] = {}

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls.register(cls)

    @classmethod
    def register(cls, sub_cls: Any):
        if hasattr(sub_cls, "datatype"):
            cls.registry[(sub_cls.chain_type, sub_cls.datatype)] = sub_cls

    @classmethod
    def from_name(cls, datatype: str, *args, **kwargs) -> Chain:
        return cls.registry[(cls.chain_type, datatype)](*args, **kwargs)


def template_names(path: str) -> list[str]:
    """
    obtain template names from a given directory

    Args:
        path (str): templates directory

    Returns:
        list[str]: a list of template names, eg: address, name, price
    #"""
    items: list[str] = os.listdir(path)

    return [os.path.splitext(path)[0] for path in items]

def auto_class(
    path: str,
    base_cls: type,
    class_suffix: str,
    generator_chain: Chain = None,
    normalizer_chain: Chain = None,
    **kwargs
) -> None:
    """
    Dynamically creating new classes for the
    generator, normalizer, and pipeline scripts

    Args:
        path (str): directory of files to obtain template names
        base_cls (type): super class to inherit from
        class_suffix (str): suffix to the generated class name
    """
    templates: list[str] = template_names(path)
    for template in templates:
        generated_type: type = type(
            template.capitalize() + class_suffix, (base_cls,), {"datatype": template}
        )

        if class_suffix == "DatasetPipeline":
            generated_type: type = type(
                template.capitalize() + class_suffix,
                (base_cls,),
                {
                    "datatype": template,
                    "generator": generator_chain.from_name(template, **kwargs),
                    "normalizer": normalizer_chain.from_name(template),
                },
            )
