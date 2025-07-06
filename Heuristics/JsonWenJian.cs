
using System;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Threading;
using System.Text.Json;
using System.Reflection.Metadata.Ecma335;
using System;
using System.Net.Http;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;
using System.Reflection.Metadata;
using System.Text.Json.Serialization;
public class JsonWenJian
{
    
    
    public int[,] JsonDaoRu(int a,string fn)
	{
        int j = 0;
        int k = 0;
        int num = a;
        string json;
        int[,] broad = new int[10, 10];
            JsonWenJian p = new JsonWenJian();
            json = fn;
        int i = 0;
        while (json[i] != '[')
        {
            i++;
        }
        char c = json[i];
            for ( ; i < json.Length; i++)
            {
                if (json[i] != ',' && json[i] != '[' && json[i] != ']')
                {
                    if (j == num)
                    {
                        j = 0;
                        k++;
                    }
                    if (json[i].Equals('-'))
                    {
                        broad[k, j] = -1;
                        i++;
                        j++;
                    }
                    else if (json[i] == '1' || json[i] == '0')
                    {
                        broad[k, j] = json[i] - '0';
                        j++;
                    }

                }
            }
        return broad;
    }

}
