using System;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Threading;
using System.Text.Json;
using System.Reflection.Metadata.Ecma335;
using static System.Formats.Asn1.AsnWriter;
public class Test
{
    struct MaxNum
    {
        public int sc;
        public int x;
        public int y;
    }
    public static void Main(string[] arg)
    {
        MaxNum maxn = new MaxNum();
        int[] dirctx = { 1, 1, 0, -1, -1, -1, 0, 1 }, dircty = { 0, 1, 1, 1, 0, -1, -1, -1 };
        int[,] broad=new int[10,10];
        int num;
        int xianshou=0;
        JsonWenJian json = new JsonWenJian();
        num = 9;
        broad = json.JsonDaoRu(num, "E:\\装json文件\\Test.json");

        for(int i = 0; i < num; i++)
        {
            for(int j = 0; j < num; j++)
            {
                xianshou += broad[i, j];
            }
        }
        EvaluateTime evaluate = new EvaluateTime();
        bool[,] Broads= new bool[10,10];
        int[,] score = new int[10,10]; 
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num; j++)
            {
                if (broad[i, j] != 0)
                {
                    for(int k = 0; k < 8; k++)
                    {
                        if (i + dirctx[k]>=0&& i + dirctx[k]<9&& j + dircty[k]>=0&& j + dircty[k]<9 && broad[i + dirctx[k], j + dircty[k]] == 0&& Broads[i + dirctx[k], j + dircty[k]] == false)
                        {
                            Broads[i + dirctx[k], j + dircty[k]] = true;
                            score[i + dirctx[k], j + dircty[k]] =evaluate.Evaluate(i + dirctx[k], j + dircty[k], broad);
                            if(score[i + dirctx[k], j + dircty[k]] > maxn.sc)
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
        Console.WriteLine(maxn.x+" "+maxn.y+xianshou);
        return;
    }
}
