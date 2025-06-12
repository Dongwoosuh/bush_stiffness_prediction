# Bush_FD_GUI.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# 1. 리소스 파일 포함
datas=[
    ('model_CAD', 'model_CAD'),  
    ('model_param', 'model_param'), 
    ('spinner.gif', '.'),
	('check.png', '.'),
]
# .ui, .gif 등 프로젝트 루트의 리소스 파일 전부 포함
datas += collect_data_files('.', includes=['*.ui', '*.gif', '*.png'])

# dgl 내부의 JSON/YAML 데이터 파일도 포함
datas += collect_data_files('dgl', includes=['*.json', '*.yaml'])

# 2. 바이너리(공유 라이브러리) 포함
binaries = []

# 3. Analysis: 스크립트, 리소스, 바이너리, 숨은 import 지정
a = Analysis(
    ['test.py'],
    pathex=['.'],    # .spec 파일 위치
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # 사용자 모듈
        'Hyundai_rc_rc',
        'Lab_rc_black_rc',
        'inference',
        'inference_CAD',
        'CATPart2step_total_RUBBER',
		'torch',
		'dgl.distributed.optim.pytorch'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
	'PySide2'
	],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# 4. PYZ (압축된 파이썬 모듈들)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# 5. EXE (onefile 빌드를 위한 설정)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BusPredictor',           # 최종 실행파일 이름
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # 콘솔창 필요 시 True
	icon='./icon.ico',
)

# 6. COLLECT (onefile 모드에서는 생략 가능하지만 명시)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='BusPredictor',
)
