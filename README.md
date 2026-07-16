import os
os.environ['OMP_NUM_THREADS'] = '1'  # 限制线程数
import json
import requests
import openpyxl
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


## 一、从网站爬取数据并清洗

# 设置请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
}
# 英雄联盟英雄列表的API地址
url = "https://game.gtimg.cn/images/lol/act/img/js/heroList/hero_list.js"
# 发送GET请求获取数据
response = requests.get(url=url, headers=headers).text
# 将JSON字符串解析为Python字典
loads = json.loads(response)
# 取出英雄列表
dic = loads['hero']

# 将爬虫得到的数据存入Excel表格
wb = openpyxl.Workbook()          # 创建一个新的工作簿
wa = wb.active                    # 获取当前活动的工作表
wa.title = "初始数据"              # 给工作表命名

# 把字典的键（即列名）作为表头写入第一行
headers = list(dic[0].keys())
wa.append(headers)

# 遍历每个英雄，把信息逐行写入Excel
for item in dic:
    heroId = item.get('heroId')
    name = item.get('name')
    alias = item.get('alias')
    title = item.get('title')
    roles = item.get('roles')
    isworkfree = item.get('isWeekFree')
    attack = item.get('attack')
    defense = item.get('defense')
    magic = item.get('magic')
    difficulty = item.get('difficulty')
    wa.append([heroId, name, alias, title, str(roles), isworkfree, attack, defense, magic, difficulty])

# 保存到当前目录
wb.save('初始数据.xlsx')

# 检查初始数据是否保存成功
file_path = '初始数据.xlsx'
print("文件是否存在:", os.path.exists(file_path))

# 读取刚才保存的Excel，进行数据清理
df = pd.read_excel('初始数据.xlsx')

# 删除不需要的列
df = df.drop(columns=['isWeekFree'])
df = df.drop(columns=['selectAudio'])
df = df.drop(columns=['banAudio'])
df = df.drop(columns=['isARAMweekfree'])
df = df.drop(columns=['ispermanentweekfree'])
df = df.drop(columns=['changeLabel'])
df = df.drop(columns=['goldPrice'])
df = df.drop(columns=['couponPrice'])
df = df.drop(columns=['camp'])
df = df.drop(columns=['campId'])
df = df.drop(columns=['keywords'])
df = df.drop(columns=['instance_id'])
df = df.drop(columns=['heroId'])
df = df.drop(columns=['name'])
df = df.drop(columns=['alias'])
df = df.drop(columns=['title'])

# 保存清理后的数据为CSV
df.to_csv("英雄联盟数据1.csv", index=False)

# roles列原本是类似 "['fighter','mage']" 的字符串列表
# 先用逗号拆分成多行（一个英雄可能属于多个职业）
df['roles'] = df['roles'].map(lambda x: x.split(','))
df_new = df.explode('roles')

# 去掉roles列中的特殊符号：单引号、中括号、空格
df_new['roles'] = df_new['roles'].str.replace("'", "")
df_new['roles'] = df_new['roles'].str.replace("[", "")
df_new['roles'] = df_new['roles'].str.replace("]", "")
df_new['roles'] = df_new['roles'].str.replace(" ", "")
df_new.to_csv("英雄联盟数据2.csv", index=False)


## 二、初始数据绘图

# 读取展开后的数据（一个英雄可能有多行，每行一个职业）
pr = pd.read_csv("英雄联盟数据2.csv")

# 统计各职业的数量
fighter = 0   # 战士
mage = 0      # 法师
tank = 0      # 坦克
assassin = 0  # 刺客
support = 0   # 辅助
marksman = 0  # 射手

# 遍历roles列，计数每个职业出现多少次
for roles in pr['roles']:
    if roles == "fighter":
        fighter += 1
    elif roles == "mage":
        mage += 1
    elif roles == "tank":
        tank += 1
    elif roles == "assassin":
        assassin += 1
    elif roles == "support":
        support += 1
    elif roles == "marksman":
        marksman += 1

# 绘制饼状图：展示各职业占比
labels = ['FIGHTER', 'MAGE', 'TANK', 'ASSASSIN', 'SUPPORT', 'MARKSMAN']
values = [fighter, mage, tank, assassin, support, marksman]
colors = ['y', 'm', 'b', 'r', 'g', 'deeppink']
explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 每块都向外突出一点

plt.title("Roles ratio", fontsize=25)
plt.pie(values, labels=labels, explode=explode, colors=colors,
        startangle=180,
        shadow=True, autopct='%1.1f%%')
plt.axis('equal')  # 保证饼图是正圆形
plt.savefig('roles_ratio.png')
plt.show()

# 统计上手难度分布（difficulty取值1~10）
df = pd.read_csv("英雄联盟数据1.csv")
df.columns = ['roles', 'attack', 'defense', 'magic', 'difficulty']

number = np.zeros(10)  # 创建长度为10的零数组，用于计数

for i in df['difficulty']:
    number[i - 1] += 1  # 难度为1的放在索引0，以此类推

# 绘制柱状图
index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
plt.title("difficulty distribution", fontsize=25)
plt.bar(index, number)
plt.savefig('diff_distribution.png')
plt.show()

## 三、决策树

roles = pd.read_csv("英雄联盟数据2.csv")
# 给每列命名，方便后续调用
roles.columns = ['roles', 'attack', 'defense', 'magic', 'difficulty']
columns = ['roles', 'attack', 'defense', 'magic', 'difficulty']

roles_mapping = {'fighter': 0, 'mage': 1, 'tank': 2, 'assassin': 3, 'support': 4, 'marksman': 5}
roles['roles'] = roles['roles'].map(roles_mapping)

# 对数据进行可视化：绘制4×4的散点图矩阵
df_data = roles.values
fig1 = plt.figure(figsize=(15, 15), dpi=60)

for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        if i == 0:
            plt.title(columns[j + 1])      # 第一行显示列名作为标题
        if j == 0:
            plt.ylabel(columns[i + 1])     # 第一列显示列名作为纵轴标签
        if i == j:
            # 对角线位置不画散点，直接显示该特征的名字
            plt.text(0.3, 0.4, columns[i + 1], fontsize=15)
            continue
        # 用第0列（roles）作为颜色，cmap='brg'给不同职业上色
        plt.scatter(df_data[:, j + 1], df_data[:, i + 1], c=df_data[:, 0], cmap='brg')

plt.tight_layout(rect=[0, 0, 1, 0.88])

# ========== 修改后的彩色标题 ==========
# 主标题 "roles" 保持黑色
plt.suptitle('roles', fontsize=20, y=0.98)

# 使用与散点图完全一致的 brg 颜色映射，为每个职业名称单独上色
cmap = plt.get_cmap('brg')
role_names = ['fighter', 'mage', 'tank', 'assassin', 'support', 'marksman']
start_x = 0.18      # 第一个职业名称的横坐标（figure坐标，范围0~1）
step_x = 0.11       # 每个名称之间的间隔

for idx, name in enumerate(role_names):
    color = cmap(idx / 5)  # 0~5 线性映射到 colormap 的 0~1
    fig1.text(start_x + idx * step_x, 0.92, name, color=color, fontsize=12, ha='center')
    # 在职业之间画黑色的分隔符 "|"
    if idx < len(role_names) - 1:
        fig1.text(start_x + idx * step_x + 0.055, 0.92, '|', color='black', fontsize=12, ha='center')
# =====================================

plt.savefig('英雄联盟数据2.png', bbox_inches='tight', pad_inches=0.0)

# 重新读取数据（因为上面的roles已经被映射成数字了）
roles1 = pd.read_csv("英雄联盟数据2.csv")

# 切分训练集和测试集（80%训练，20%测试）
all_inputs = roles1[['attack', 'defense', 'magic', 'difficulty']].values
all_classes = roles1['roles'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    all_inputs, all_classes, train_size=0.8, random_state=1)

# 创建决策树分类器，使用信息增益（entropy）作为划分标准
dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=8)
# 训练模型
model = dtc.fit(X_train, Y_train)

# 输出模型准确率
print("测试集准确率:", dtc.score(X_test, Y_test))
print("训练集准确率:", dtc.score(X_train, Y_train))
print("全部数据准确率:", dtc.score(all_inputs, all_classes))

# 决策树可视化
font2 = {'weight': 'normal', 'size': 20}
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

fig2 = plt.figure(figsize=(20, 20))
tree.plot_tree(dtc, filled=True,
               feature_names=['attack', 'defense', 'magic', 'difficulty'],
               class_names=['fighter', 'mage', 'tank', 'assassin', 'support', 'marksman'])

plt.savefig('roles_dtc.png', bbox_inches='tight', pad_inches=0.0, dpi=600)
plt.show()


## 四、聚类前数据分析

sns.set_style("whitegrid")

# 导入数据集，只取前5列（roles + 4个数值特征）
lol = pd.read_csv("英雄联盟数据2.csv", usecols=[0, 1, 2, 3, 4])
# 设置颜色主题（antV配色）
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2']

# 绘制小提琴图（Violinplot），展示每个职业在各属性上的分布
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)
sns.violinplot(x='roles', y='attack', hue='roles', data=lol, palette=antV, ax=axes[0, 0], legend=False)
sns.violinplot(x='roles', y='defense', hue='roles', data=lol, palette=antV, ax=axes[0, 1], legend=False)
sns.violinplot(x='roles', y='magic', hue='roles', data=lol, palette=antV, ax=axes[1, 0], legend=False)
sns.violinplot(x='roles', y='difficulty', hue='roles', data=lol, palette=antV, ax=axes[1, 1], legend=False)
plt.savefig('violin.png')
plt.show()

# 绘制点图（pointplot），展示每个职业在各属性上的均值变化
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)
sns.pointplot(x='roles', y='attack', hue='roles', data=lol, palette=antV, ax=axes[0, 0], legend=False)
sns.pointplot(x='roles', y='defense', hue='roles', data=lol, palette=antV, ax=axes[0, 1], legend=False)
sns.pointplot(x='roles', y='magic', hue='roles', data=lol, palette=antV, ax=axes[1, 0], legend=False)
sns.pointplot(x='roles', y='difficulty', hue='roles', data=lol, palette=antV, ax=axes[1, 1], legend=False)
plt.savefig('point.png')
plt.show()

## 五、确定K-means的最佳K值

lol = pd.read_csv("英雄联盟数据1.csv")
# 去掉类别列roles，只保留数值特征用于聚类
data = lol.drop(labels='roles', axis=1)

# 使用MinMaxScaler将数据缩放到0~1之间，消除量纲影响
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

# 手肘法：尝试不同的聚类个数K（从2到14）
distortions = []   # 簇内误差平方和（SSE）
sil_score = []     # 轮廓系数

for i in range(2, 15):
    kmeans_model = KMeans(n_clusters=i)
    predict_y = kmeans_model.fit_predict(data)

    distortions.append(kmeans_model.inertia_)           # 记录SSE
    sil_score.append(silhouette_score(data, predict_y))  # 记录轮廓系数

print('簇内误差平方和：', distortions)
print('轮廓系数：', sil_score)

# 绘制手肘法曲线
plt.plot(range(2, 15), distortions, marker='x')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('distortions')
plt.savefig('distortions.png')
plt.show()

# 绘制轮廓系数曲线
plt.plot(range(2, 15), sil_score, marker='x')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.title('silhouette_score')
plt.savefig('silhouette_score.png')
plt.show()


## 六、K-means聚类效果

# 读取展开后的数据（含roles列）
lol = pd.read_csv("英雄联盟数据2.csv")
print(lol.head())  # 查看前几行

# 提取建模用的特征X（去掉roles列）
X = lol.drop(labels='roles', axis=1)

# 构建KMeans模型，指定聚为4类
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 将聚类结果作为新列加入X
X['cluster'] = kmeans.labels_
# 统计每个簇有多少个样本
a = X.cluster.value_counts()
print(a)

# 获取4个簇的中心点坐标
centers = kmeans.cluster_centers_

# 绘制聚类散点图：attack vs magic
sns.lmplot(x='attack', y='magic', hue='cluster', markers=['^', 'v', '<', '>'],
           data=X, fit_reg=False, scatter_kws={'alpha': 0.8}, facet_kws={"legend_out": False})
# 在图上标出簇中心
plt.scatter(centers[:, 0], centers[:, 2], marker='*', color='black', s=50)
plt.xlabel('attack')
plt.ylabel('magic')

plt.savefig('kmeans.png')
plt.show()
