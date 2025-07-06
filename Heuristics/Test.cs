using System;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Threading;
using System.Text.Json;
using System.Reflection.Metadata.Ecma335;
using static System.Formats.Asn1.AsnWriter;
using Newtonsoft.Json;
using GomokuApiLibrary;
public class Test
{
    static bool nextone = true;
    static int ids = 3;










    struct MaxNum
    {
        public int sc;
        public int x;
        public int y;
    }
    static bool KongBai(bool[,] a,int num)
    {
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num; j++) {
                if (a[i, j] == true)
                {
                    return false;
                }
            }
        }
                return true;
    }
    public static async Task Main(string[] arg)
    {
        var apiClient = new GomokuApiClient("https://gomoku.handywote.site/");

        // 创建新的棋盘
        var createResponse = await apiClient.CreateChessboard(new GomokuRequest { id = ids,x = 9 ,y = 9 });
        Console.WriteLine($"Create Response: {JsonConvert.SerializeObject(createResponse)}");

        // 更新棋盘数据
        //var updateResponse = await apiClient.UpdateChessboard(new GomokuRequest { id = ids, x = 2, y = 1, player = 1 });

        // 获取棋盘数据
        var getResponse = await apiClient.GetChessboard(ids,true);
        //Console.WriteLine($"Get Response: {JsonConvert.SerializeObject(getResponse)}");
        string st;
        st = JsonConvert.SerializeObject(getResponse);
        char first = st[34];
        while (st[50] == 'f'|| st[49] == 'f')
        {
            getResponse = await apiClient.GetChessboard(ids, true);
            st = JsonConvert.SerializeObject(getResponse);
            if (st[34] == first)
            {
                MaxNum maxn = new MaxNum();
                int[] dirctx = { 1, 1, 0, -1, -1, -1, 0, 1 }, dircty = { 0, 1, 1, 1, 0, -1, -1, -1 };
                int[,] broad = new int[10, 10];
                int num;
                int xianshou = 0;
                JsonWenJian json = new JsonWenJian();
                num = 9;

                broad = json.JsonDaoRu(num, JsonConvert.SerializeObject(getResponse));

                for (int i = 0; i < num; i++)
                {
                    for (int j = 0; j < num; j++)
                    {
                        xianshou += broad[i, j];
                    }
                }

                EvaluateTime evaluate = new EvaluateTime();
                bool[,] Broads = new bool[10, 10];
                int[,] score = new int[10, 10];
                for (int i = 0; i < num; i++)
                {
                    for (int j = 0; j < num; j++)
                    {
                        if (broad[i, j] != 0)
                        {
                            for (int k = 0; k < 8; k++)
                            {
                                if (i + dirctx[k] >= 0 && i + dirctx[k] < 9 && j + dircty[k] >= 0 && j + dircty[k] < 9 && broad[i + dirctx[k], j + dircty[k]] == 0 && Broads[i + dirctx[k], j + dircty[k]] == false)
                                {
                                    Broads[i + dirctx[k], j + dircty[k]] = true;
                                    score[i + dirctx[k], j + dircty[k]] = evaluate.Evaluate(i + dirctx[k], j + dircty[k], broad);
                                    if (score[i + dirctx[k], j + dircty[k]] > maxn.sc)
                                    {
                                        maxn.sc = score[i + dirctx[k], j + dircty[k]];
                                        maxn.x = i + dirctx[k];
                                        maxn.y = j + dircty[k];
                                    }
                                    else if (score[i + dirctx[k], j + dircty[k]] == maxn.sc)
                                    {
                                        if (Math.Pow(maxn.x - 4, 2) + Math.Pow(maxn.y - 4, 2) > Math.Pow(i + dirctx[k] - 4, 2) + Math.Pow(j + dircty[k] - 4, 2))
                                        {
                                            maxn.sc = score[i + dirctx[k], j + dircty[k]];
                                            maxn.x = i + dirctx[k];
                                            maxn.y = j + dircty[k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (Test.KongBai(Broads, num))
                    {
                        maxn.x = 4;
                        maxn.y = 4;
                    }
                }
                if (xianshou == 0) { xianshou = 1; } else { xianshou = -1; }
                var updateResponse = await apiClient.UpdateChessboard(new GomokuRequest { id = ids, x = maxn.x, y = maxn.y, player = xianshou });
                Console.WriteLine(maxn.x + " " + maxn.y + " " + xianshou);
            }
        }
       
        return ;
    }
}
