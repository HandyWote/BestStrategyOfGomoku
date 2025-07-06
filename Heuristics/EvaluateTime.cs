
using System;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Threading;
using System.Text.Json;
using System.Reflection.Metadata.Ecma335;
public class EvaluateTime
{
	int[] dirctx = { 1, 1, 0, -1, -1, -1, 0, 1 }, dircty = { 0, 1, 1, 1, 0, -1, -1, -1 };
    int erlian = 8, sanlian = 100, silian = 10000, wulian = 100000, danmianshouzuerlian = 2, danmianshouzusanlian = 10, danmianshouzusilian = 100;
    public int Evaluates(int a,int z)
	{
		int num=0;
		if (a == 2&&z==0)
		{
            num += erlian;
		}
		if (a == 2&&z==1)
		{
            num += danmianshouzuerlian;

        }
        if (a == 3&&z==0)
        {
            num += sanlian;
        }
        if (a == 3&&z==1)
        {
            num += danmianshouzusanlian;
        }
        if (a == 4&&z==0)
        {
            num += silian;
        }
        if (a == 4&&z==1)
        {
            num += danmianshouzusilian;
        }
        if(a >= 5)
        {
            num += wulian;
        }
        return num;
    }
	public EvaluateTime()
	{
		
	}
    public int Evaluate(int x,int y, int[,] broad)
	{
		int score_b=0,score_w=0;
        for (int i = 0; i < 4; i++)
		{
			int num=1;
			int zudang = 0;
			int x1 = x, y1 = y;
                while (x1 + dirctx[i] >= 0 && x1 + dirctx[i] < 9 && y1 + dircty[i] >= 0 && y1 + dircty[i] < 9&&broad[x1 + dirctx[i], y1 + dircty[i]] == 1)
			{
				x1 += dirctx[i]; y1 += dircty[i];
				num++;
			}
			if(x1 + dirctx[i] >= 0 && x1 + dirctx[i] < 9 && y1 + dircty[i] >= 0 && y1 + dircty[i] < 9)
			{
                if (broad[x1 + dirctx[i], y1 + dircty[i]] == -1)
                {
                    zudang++;
                }

			}
            else
            {
                zudang++;
            }
			x1 = x; y1 = y;
            while (x1 + dirctx[i + 4] >= 0 && x1 + dirctx[i + 4] < 9 && y1 + dircty[i + 4] >= 0 && y1 + dircty[i + 4] < 9&&broad[x1 + dirctx[i+4], y1 + dircty[i+4]] == 1)
            {
                x1 += dirctx[i+4]; y1 += dircty[i+4];
                num++;
            }
            if (x1 + dirctx[i + 4] >= 0 && x1 + dirctx[i + 4] < 9 && y1 + dircty[i + 4] >= 0 && y1 + dircty[i + 4] < 9)
            {
                if (broad[x1 + dirctx[i + 4], y1 + dircty[i + 4]] == -1)
                {
                    zudang++;
                }
            }
            else
            {
                zudang++;
            }
            score_b += Evaluates(num, zudang);
        }
        for (int i = 0; i < 4; i++)
        {
            int num = 1;
            int zudang = 0;
            int x1 = x, y1 = y;
            while (x1 + dirctx[i] >= 0 && x1 + dirctx[i] < 9 && y1 + dircty[i] >= 0 && y1 + dircty[i] < 9 && broad[x1 + dirctx[i], y1 + dircty[i]] == -1)
            {
                x1 += dirctx[i]; y1 += dircty[i];
                num++;
            }
            if (x1 + dirctx[i] >= 0 && x1 + dirctx[i] < 9 && y1 + dircty[i] >= 0 && y1 + dircty[i] < 9)
            {
                if (broad[x1 + dirctx[i], y1 + dircty[i]] == 1)
                {
                    zudang++;
                }
            }
            else
            {
                zudang++;
            }
            x1 = x; y1 = y;
            while (x1 + dirctx[i + 4] >= 0 && x1 + dirctx[i + 4] < 9 && y1 + dircty[i + 4] >= 0 && y1 + dircty[i + 4] < 9 && broad[x1 + dirctx[i + 4], y1 + dircty[i + 4]] == -1)
            {
                x1 += dirctx[i + 4]; y1 += dircty[i + 4];
                num++;
            }
            if (x1 + dirctx[i + 4] >= 0 && x1 + dirctx[i + 4] < 9 && y1 + dircty[i + 4] >= 0 && y1 + dircty[i] < 9)
            {
                if (broad[x1 + dirctx[i + 4], y1 + dircty[i + 4]] == 1)
                {
                    zudang++;
                }
            }
            else
            {
                zudang++;
            }
            score_w += Evaluates(num, zudang);
        }
        if(score_b<score_w) score_b = score_w;
        return score_b;
	}
}
