实验数据 ：kaggle的60个比赛 每一个比赛选取一个作者的代码  数据输入规模从6.7MB到14.3GB<kbd>kaggle_code.csv</dbd>

实验结果：1. 加入类型和维度信息的 ALTERAPI工具生成 387<kbd>type_candidate.csv</dbd>个候选的可替代实现，经输入输出相等验证，其中218<kbd>optimizetion.csv</dbd> 个可替代实现验证为真 
            其中79个可替代实现减速（79/218）36% 剩下的为提速 64%
          2.不加类型和维度信息的 ALTERAPI工具生成 667<kbd>no_type_candidate.csv</dbd>个候选的可替代实现，经输入输出相等验证，其中218<kbd>optimizetion.csv</dbd>个可替代实现验证为真
             其中79个可替代实现减速（79/218）36% 剩下的为提速 64%
           
实验结论：在APTERAPI中，加入变量的类型和维度信息可以减少候选集，提高高效API推荐算法的准确率
