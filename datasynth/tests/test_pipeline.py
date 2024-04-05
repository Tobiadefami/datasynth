import pytest
from datasynth.pipelines import DatasetPipeline



@pytest.mark.parametrize("datatype", ["name", "address", "price"])
def test_pipeline(
    datatype: str 
):
    chain = DatasetPipeline.from_name(datatype, k=1, dataset_name="test_dataset")
    # No-op thing is a hack, not sure why it won't let me run with no args
    results = chain.run(noop="true")
    assert len(results['outputs']) == 1
    assert len(results['outputs'][0]['input']) > 0 
    assert isinstance(results['outputs'][0]['output'], list)
    assert isinstance(results['outputs'][0]['output'][0], dict)


