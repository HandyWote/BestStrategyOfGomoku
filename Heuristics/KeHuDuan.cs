using Newtonsoft.Json;
using System.Text;

namespace GomokuApiLibrary
{
    public class GomokuApiClient
    {
        private readonly HttpClient _httpClient;

        public GomokuApiClient(string baseUrl)
        {
            _httpClient = new HttpClient { BaseAddress = new Uri(baseUrl) };
        }

        public async Task<GomokuResponse> CreateChessboard(GomokuRequest request)
        {
            var content = new StringContent(JsonConvert.SerializeObject(request), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("api/gomoku", content);
            response.EnsureSuccessStatusCode();
            return JsonConvert.DeserializeObject<GomokuResponse>(await response.Content.ReadAsStringAsync());
        }

        public async Task<GomokuResponse> GetChessboard(int id, bool showStatus = false)
        {
            var url = $"api/gomoku/{id}";
            if (showStatus)
            {
                url += "?showStatus=true";
            }
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            return JsonConvert.DeserializeObject<GomokuResponse>(await response.Content.ReadAsStringAsync());
        }

        public async Task<GomokuResponse> UpdateChessboard(GomokuRequest request)
        {
            var content = new StringContent(JsonConvert.SerializeObject(request), Encoding.UTF8, "application/json");
            var response = await _httpClient.PutAsync($"api/gomoku/{request.id}", content);
            response.EnsureSuccessStatusCode();
            return JsonConvert.DeserializeObject<GomokuResponse>(await response.Content.ReadAsStringAsync());
        }

        public async Task<string> DeleteChessboard(int id)
        {
            var response = await _httpClient.DeleteAsync($"api/gomoku/{id}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }

    public class GomokuRequest
    {
        public int id { get; set; }
        public int? x { get; set; }
        public int? y { get; set; }
        public int? player { get; set; }
        public int[][] board { get; set; }
    }

    public class GomokuResponse
    {
        public int code { get; set; }
        public string msg { get; set; }
        public int nextPlayer { get; set; }
        public bool isGameOver { get; set; }
        public int winner { get; set; }
        public int[][] board { get; set; }
    }
}