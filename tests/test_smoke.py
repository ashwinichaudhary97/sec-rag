from sec_rag import main


def test_main(capsys):
    main()
    assert capsys.readouterr().out == "Hello from sec-rag!\n"
