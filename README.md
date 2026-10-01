<<<<<<< HEAD
# Benchmark comparativo de federated unlearning

Esta pasta reúne o benchmark e snapshots locais dos códigos de FedOSD, FedUP e Fast-FedUL. O repositório nativo do Maverick fica fora desta pasta; a adaptação Maverick usada na comparação comum está em `common/methods.py`.

As pastas originais dos projetos continuam na raiz do workspace com seus históricos Git independentes. Os snapshots em `repositories/` incluem o estado de trabalho atual, sem ambientes virtuais, datasets, checkpoints ou históricos Git aninhados. `SOURCE_REVISION` registra o commit de origem de cada snapshot.

## Executar o benchmark

No WSL, entre nesta pasta:

```bash
cd /mnt/e/Federated-unlearning/comparative_benchmark
bash setup_uv_wsl.sh common
uv run --locked python scripts/run_experiment.py --config configs/pilot_common.json --phase all --methods all --device cpu --force
uv run --locked python scripts/aggregate_results.py --results results/pilot_common_v2 --require-all --require-seeds 1
```

O piloto usa 600 imagens e duas rodadas para conferir o pipeline. Para os experimentos maiores, ajuste os arquivos em `configs/` e use os mesmos scripts. O CIFAR-10 é baixado automaticamente pelo torchvision quando necessário.

Para preparar ambientes dos projetos nativos incluídos:

```bash
bash setup_uv_wsl.sh fedosd fedup fastfedul
```

O benchmark comum executa adaptações comparáveis em `common/methods.py`; os ambientes nativos servem para reproduções específicas dos projetos.

## Publicar no GitHub

`data/`, `results/`, ambientes virtuais, caches, logs e checkpoints locais são ignorados pelo `.gitignore`. Assim, os datasets e saídas experimentais não entram no envio padrão. As configurações e o código para reproduzir os resultados permanecem incluídos.
=======
# Federated-Unlearnig_Comparative
Comparative models Federated Unlearning
>>>>>>> 1773049813132966383b012cd1b0d1e103842567
