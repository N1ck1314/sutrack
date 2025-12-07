# 本地配置文件设置说明

## 首次使用须知

在首次克隆本项目后，您需要创建两个本地配置文件来设置您的数据集路径和工作目录。

### 1. 训练配置文件

**文件位置**: `lib/train/admin/local.py`

**操作步骤**:
```bash
# 复制示例文件
cp lib/train/admin/local.py.example lib/train/admin/local.py

# 编辑文件，修改为您本地的实际路径
vim lib/train/admin/local.py  # 或使用您喜欢的编辑器
```

### 2. 测试配置文件

**文件位置**: `lib/test/evaluation/local.py`

**操作步骤**:
```bash
# 复制示例文件
cp lib/test/evaluation/local.py.example lib/test/evaluation/local.py

# 编辑文件，修改为您本地的实际路径
vim lib/test/evaluation/local.py  # 或使用您喜欢的编辑器
```

### 3. 需要修改的路径

请将示例文件中的 `/path/to/your/SUTrack` 和 `/path/to/dataset/` 替换为您的实际路径。

例如：
- `workspace_dir`: SUTrack项目的根目录
- `got10k_dir`: GOT-10k数据集的路径
- `lasot_dir`: LaSOT数据集的路径
- 其他数据集路径根据您的需要配置

### 4. 重要提示

⚠️ **`local.py` 文件已被添加到 `.gitignore`，不会被提交到git仓库中。**

这样做是为了：
- 保护您的本地路径隐私
- 避免不同机器之间的路径冲突
- 每台机器可以使用自己的配置

### 5. 验证配置

配置完成后，可以运行以下命令验证：
```bash
python -c "from lib.train.admin.local import EnvironmentSettings; env = EnvironmentSettings(); print(env.workspace_dir)"
```

如果输出了您设置的路径，说明配置成功！
