from providence import DataResolver, DataSource


def test_pipeline_retains_data_source(tmp_path):
    data = DataResolver(
            {
            "x": [
                [1., 2.],
                [3., 4.],
            ],
        },
        cache_dir=tmp_path,
    )

    pipeline()

    assert isinstance(data.data_source, DataSource)
    assert data.data_source.uri is None
    assert data.data_source.revision is None
    assert data.data_source.checksum is not None
    assert len(data.data_source.checksum) == 64
