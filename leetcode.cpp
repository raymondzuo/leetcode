#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <set>
#include <stack>
#include <unordered_set>
#include <climits>
#include <unordered_map>
#include <stdio.h>
#include <sstream>
#include <deque>
#include <list>

using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::max;
using std::pair;
using std::priority_queue;
using std::queue;
using std::set;
using std::stack;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::deque;
using std::list;

/**
 * leetcode most frequency problems
 * 
 */

//#1 Two sum
vector<int> twoSum(vector<int> &nums, int target)
{
    vector<int> result;
    std::map<int, int> mymap;

    for (int i = 0; i < nums.size(); i++)
    {
        int another = target - nums[i];
        auto it = mymap.find(another);
        if (it != mymap.end())
        {
            result.push_back(it->second);
            result.push_back(i);
            return result;
        }

        mymap[nums[i]] = i;
    }
}

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
//#2 Add two numbers
ListNode *addTwoNumbers(ListNode *l1, ListNode *l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    int carry = 0;
    ListNode *p1 = l1, *p2 = l2;

    while (p1 || p2)
    {
        int x = p1 ? p1->val : 0;
        int y = p2 ? p2->val : 0;

        int sum = x + y + carry;
        carry = sum / 10;
        sum = sum % 10;
        ListNode *temp = new ListNode(sum);

        cur->next = temp;
        cur = temp;

        if (p1)
            p1 = p1->next;
        if (p2)
            p2 = p2->next;
    }

    if (carry > 0)
        cur->next = new ListNode(carry);

    return head->next;
}

/********************/
//#200. Number of Islands
void dfs(vector<vector<char>> &grid, int r, int c)
{
    int maxr = grid.size();
    int maxc = grid[0].size();

    if (r < 0 || c < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == '0')
        return;
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}

int numIslands(vector<vector<char>> &grid)
{
    int count = 0;

    int maxr = grid.size();
    if (0 == maxr)
        return count;
    int maxc = grid[0].size();
    if (0 == maxc)
        return count;

    for (int i = 0; i < maxr; i++)
    {
        for (int j = 0; j < maxc; j++)
        {
            if ('1' == grid[i][j])
            {
                count++;
                dfs(grid, i, j);
            }
        }
    }

    return count;
}

int numIslandsBFS(vector<vector<char>> &grid)
{
    int nr = grid.size();
    if (0 == nr)
        return 0;
    int nc = grid[0].size();

    int count = 0;
    for (int r = 0; r < nr; r++)
    {
        for (int c = 0; c < nc; c++)
        {
            if ('1' == grid[r][c])
            {
                count++;
                grid[r][c] = '0';
                queue<pair<int, int>> tempqueue;
                tempqueue.push({r, c});

                while (!tempqueue.empty())
                {
                    auto front = tempqueue.front();
                    tempqueue.pop();
                    int tempr = front.first;
                    int tempc = front.second;

                    if (tempr - 1 >= 0 && grid[tempr - 1][tempc] == '1')
                    {
                        tempqueue.push({tempr - 1, tempc});
                        grid[tempr - 1][tempc] = '0';
                    }
                    if (tempr + 1 < nr && grid[tempr + 1][tempc] == '1')
                    {
                        tempqueue.push({tempr + 1, tempc});
                        grid[tempr + 1][tempc] = '0';
                    }
                    if (tempc - 1 >= 0 && grid[tempr][tempc - 1] == '1')
                    {
                        tempqueue.push({tempr, tempc - 1});
                        grid[tempr][tempc - 1] = '0';
                    }
                    if (tempc + 1 < nc && grid[tempr][tempc + 1] == '1')
                    {
                        tempqueue.push({tempr, tempc + 1});
                        grid[tempr][tempc + 1] = '0';
                    }
                }
            }
        }
    }

    return count;
}

/********************/

/********************/
//#5. Longest Palindromic Substring
string longestPalindrome(string s)
{
    int n = s.length();
    int max = 1;
    int si = 0;   //start
    int dp[n][n]; //n->n from n to n
    //length is 1
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                dp[i][j] = 1;
            else
                dp[i][j] = 0;
        }
    }
    //length is 2
    for (int i = 0; i < n - 1; i++)
    {
        if (s[i + 1] == s[i])
        {
            dp[i][i + 1] = 1;
            max = 2;
            si = i;
        }
    }

    //length large than 3
    for (int k = 3; k <= n; k++)
    {
        for (int i = 0; i < n - k + 1; i++)
        {
            int j = i + k - 1;
            if (dp[i + 1][j - 1] && s[i] == s[j])
            {
                dp[i][j] = 1;
                if (k > max)
                {
                    max = k;
                    si = i;
                }
            }
        }
    }

    return s.substr(si, max);
}

/********************/
/* 3. Longest Substring Without Repeating Characters*/
int lengthOfLongestSubstring(string s)
{
    int max = 0, start = 0;
    vector<char> dp(256, -1);

    for (int i = 0; i < s.length(); i++)
    {
        start = std::max(dp[s[i]] + 1, start);
        dp[s[i]] = i;
        max = std::max(max, i - start + 1);
    }

    return max;
}

/********************/
/* 15. 3Sum */
vector<vector<int>> threeSum(vector<int> &num)
{
    std::sort(num.begin(), num.end());
    vector<vector<int>> result;

    for (int i = 0; i < num.size(); i++)
    {
        int target = -num[i];
        int left = i + 1;
        int right = num.size() - 1;
        while (left < right)
        {
            int sum = num[left] + num[right];
            if (sum > target)
            {
                right--;
            }
            else if (sum < target)
            {
                left++;
            }
            else
            {
                vector<int> temp;
                temp.push_back(num[i]);
                temp.push_back(num[left]);
                temp.push_back(num[right]);
                result.push_back(temp);

                while (left < right && num[left] == temp[1])
                {
                    left++;
                }
                while (left < right && num[right] == temp[2])
                {
                    right--;
                }
            }
        }

        while (i + 1 < num.size() && num[i + 1] == num[i])
            i++;
    }

    return result;
}
/********************/
/* 20. Valid Parentheses */
bool isValid(string s)
{
    std::map<char, char> match_map;
    match_map['('] = ')';
    match_map['{'] = '}';
    match_map['['] = ']';

    stack<char> mystack;
    for (char c : s)
    {
        if (match_map.find(c) != match_map.end())
        {
            mystack.push(c);
        }
        else
        {
            if (mystack.empty())
                return false;
            char matchChar = mystack.top();
            if (c != match_map[matchChar])
                return false;

            mystack.pop();
        }
    }

    return mystack.empty();
}
/********************/
/* 56. Merge Intervals */
struct Interval
{
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

vector<Interval> merge(vector<Interval> &intervals)
{
    std::sort(intervals.begin(), intervals.end(), [&](struct Interval a, struct Interval b) {
        return a.start < b.start;
    });

    for (auto it = intervals.begin(); it != intervals.end();)
    {
        if ((it + 1) != intervals.end() && (it + 1)->start <= it->end)
        {
            int nextEnd = (it + 1)->end;
            it->end = nextEnd > it->end ? nextEnd : it->end;
            intervals.erase(it + 1);
        }
        else
        {
            it++;
        }
    }

    return intervals;
}

/********************/
/* 21. Merge Two Sorted Lists */
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
{
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    ListNode *p1 = l1, *p2 = l2;

    while (p1 && p2)
    {
        cur->next = p1->val > p2->val ? p2 : p1;
        if (p1->val > p2->val)
        {
            cur->next = p2;
            p2 = p2->next;
        }
        else
        {
            cur->next = p1;
            p1 = p1->next;
        }

        cur = cur->next;
    }

    cur->next = p1 ? p1 : p2;

    return head->next;
}

/********************/
/* 53. Maximum Subarray */
int maxSubArray(vector<int> &nums)
{
    int n = nums.size();
    int dp[n];
    for (int i = 0; i < n; i++)
        dp[i] = -INT64_MAX;

    dp[0] = nums[0];
    int max = nums[0];
    for (int i = 1; i < n; i++)
    {
        dp[i] = dp[i - 1] > 0 ? dp[i - 1] + nums[i] : nums[i];
        if (dp[i] > max)
            max = dp[i];
    }

    return max;
}

/********************/
/* 253. Meeting Rooms II */
int minMeetingRooms(vector<Interval> &intervals)
{
    std::map<int, int> mymap;
    for (auto &i : intervals)
    {
        mymap[i.start]++;
        mymap[i.end]--;
    }

    int max = 0;
    int temp = 0;
    for (auto it = mymap.begin(); it != mymap.end(); it++)
    {
        temp += it->second;
        if (temp > max)
            max = temp;
    }

    return max;
}

/********************/
/* 206. Reverse Linked List*/
ListNode *reverseList(ListNode *head)
{
    ListNode *prev = NULL;
    ListNode *next = NULL;

    while (head)
    {
        next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }

    return prev;
}

/********************/
/* 121. Best Time to Buy and Sell Stock*/
int maxProfit(vector<int> &prices)
{
    if (prices.size() == 0)
        return 0;
    int min = prices[0];
    int maxProfit = 0;
    for (int i = 1; i < prices.size(); i++)
    {
        if (prices[i] < min)
            min = prices[i];

        if (prices[i] - min > maxProfit)
            maxProfit = prices[i] - min;
    }
    return maxProfit;
}

/********************/
/* 7. Reverse Integer */
int reverse(int x)
{
    long long target = 0;
    while (x)
    {
        target = (x % 10) + target * 10;
        x /= 10;
    }

    return (target > INT32_MAX || target < INT32_MIN) ? 0 : target;
}
/********************/
/* 238. Product of Array Except Self*/
vector<int> productExceptSelf(vector<int> &nums)
{
    int n = nums.size();
    int front[n], back[n];
    front[0] = nums[0];
    back[0] = nums[n - 1];

    for (int i = 1; i < n; i++)
    {
        front[i] = front[i - 1] * nums[i];
        back[i] = back[i - 1] * nums[n - i - 1];
    }

    vector<int> result;
    for (int i = 0; i < n; i++)
    {
        int allFront = i - 1 >= 0 ? front[i - 1] : 1;
        int allBack = i + 1 < n ? back[n - 2 - i] : 1;
        result.push_back(allFront * allBack);
    }

    return result;
}

/********************/
/* 973. K Closest Points to Origin*/
vector<vector<int>> kClosest(vector<vector<int>> &points, int K)
{
    std::sort(points.begin(), points.end(), [&](vector<int> &a, vector<int> &b) {
        return (pow(a[0], 2) + pow(a[1], 2)) < (pow(b[0], 2) + pow(b[1], 2));
    });

    return vector<vector<int>>(points.begin(), points.begin() + K);
}

/********************/
/* 11. Container With Most Water*/
int maxArea(vector<int> &height)
{
    int s = 0, e = height.size() - 1;
    int max = 0;

    while (s < e)
    {
        int len = e - s;
        int min = 0;
        if (height[s] < height[e])
        {
            min = height[s];
            s++;
        }
        else
        {
            min = height[e];
            e--;
        }

        int temp_area = len * min;
        if (temp_area > max)
            max = temp_area;
    }

    return max;
}

/********************/
/* 33. Search in Rotated Sorted Array */
int search_rotate_index(vector<int> &arr, int low, int high)
{
    if (arr[low] < arr[high])
        return 0;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] > arr[mid + 1])
        {
            return mid + 1;
        }
        else
        {
            if (arr[mid] >= arr[low])
                low = mid + 1;
            else
            {
                high = mid - 1;
            }
        }
    }
}

int binary_search(vector<int> &arr, int low, int high, int target)
{
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] == target)
            return mid;
        else
        {
            if (arr[mid] < target)
            {
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }
    }

    return -1;
}

int search(vector<int> &arr, int target)
{
    int n = arr.size();
    if (n == 0)
        return -1;
    if (n == 1)
        return arr[0] == target ? 0 : -1;
    int rotate_index = search_rotate_index(arr, 0, n - 1);
    if (arr[rotate_index] == target)
        return rotate_index;
    if (rotate_index == 0)
        return binary_search(arr, 0, n - 1, target);

    if (target < arr[0])
        return binary_search(arr, rotate_index, n - 1, target);
    else
        return binary_search(arr, 0, rotate_index - 1, target);
}

/********************/
/* 102. Binary Tree Level Order Traversal*/
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

vector<vector<int>> levelOrder(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;
    queue<TreeNode *> myqueue;
    myqueue.push(root);

    while (!myqueue.empty())
    {
        vector<int> level_vec;
        int len = myqueue.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *front = myqueue.front();
            level_vec.push_back(front->val);
            myqueue.pop();

            if (NULL != front->left)
                myqueue.push(front->left);
            if (NULL != front->right)
                myqueue.push(front->right);
        }

        result.push_back(level_vec);
    }

    return result;
}

void levelOrderHelper(vector<vector<int>> &result, vector<TreeNode> &nodes)
{
    if (nodes.size() == 0)
        return;

    vector<int> levelNodes;
    vector<TreeNode> nexLevelNodes;

    for (TreeNode node : nodes)
    {
        levelNodes.push_back(node.val);
        if (node.left != NULL)
            nexLevelNodes.push_back(*node.left);
        if (node.right != NULL)
            nexLevelNodes.push_back(*node.right);
    }
    result.push_back(levelNodes);

    levelOrderHelper(result, nexLevelNodes);
}

vector<vector<int>> levelOrderRecursion(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;

    vector<TreeNode> init;
    init.push_back(*root);

    levelOrderHelper(result, init);

    return result;
}
/********************/
/* 22. Generate Parentheses */
void back_track(vector<string> &result, string &strTemp, int open, int close, int max)
{
    if (strTemp.length() == 2 * max)
    {
        result.push_back(strTemp);
        strTemp.clear();
        return;
    }

    if (open < max)
        back_track(result, strTemp.append("("), open + 1, close, max);
    if (close < open)
        back_track(result, strTemp.append(")"), open, close + 1, max);
}

vector<string> generateParenthesis(int n)
{
    vector<string> result;
    string strTemp;
    back_track(result, strTemp, 0, 0, n);

    return result;
}

/********************/
/* 771. Jewels and Stones*/
int numJewelsInStones(string J, string S)
{
    int count = 0;
    unordered_set<char> myset;
    for (char c : J)
        myset.insert(c);
    for (char c : S)
    {
        if (myset.find(c) != myset.end())
        {
            count++;
        }
    }

    return count;
}

/********************/
/* 141. Linked List Cycle*/
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

bool hasCycle(ListNode *head)
{
    if (head == NULL || head->next == NULL)
        return false;

    ListNode *slow = head, *fast = head->next;
    while (slow != NULL && fast != NULL)
    {
        if (slow == fast)
            return true;
        slow = slow->next;
        fast = fast->next;
        if (fast != NULL)
            fast = fast->next;
    }

    return false;
}

/********************/
/* 9. Palindrome Number */
bool isPalindrome(int x)
{
    if (x < 0)
        return false;

    long long temp = 0;
    int tempx = x;
    while (x != 0)
    {
        temp = x % 10 + temp * 10;
        x /= 10;
    }
    if (temp > INT_MAX)
        temp = 0;
    return tempx == temp;
}

/********************/
/* 322. Coin Change*/
int coinChange(vector<int> &coins, int amount)
{
    int dp[amount + 1];
    for (int i = 0; i <= amount; i++)
        dp[i] = amount + 1;

    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
    {
        for (int j = 0; j < coins.size(); j++)
        {
            if (i >= coins[j])
                dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}

/********************/
/* 560. Subarray Sum Equals K */
int subarraySum(vector<int> &nums, int k)
{
    int count = 0;
    int n = nums.size();
    int sum[n + 1];
    sum[0] = 0;
    for (int i = 1; i <= nums.size(); i++)
        sum[i] = sum[i - 1] + nums[i - 1];

    for (int i = 0; i <= n; i++)
    {
        for (int j = i + 1; j <= n; j++)
        {
            if (sum[j] - sum[i] == k)
                count++;
        }
    }

    return count;
}

int subarraySumHash(vector<int> &nums, int k)
{
    unordered_map<int, int> sum_map;
    int sum = 0;
    int count = 0;
    sum_map[0] = 1;

    for (int i = 1; i <= nums.size(); i++)
    {
        sum += nums[i - 1];
        count += sum_map[sum - k];
        ++sum_map[sum];
    }

    return count;
}
/********************/
/* 560. Subarray Sum Equals K */
vector<int> spiralOrder(vector<vector<int>> &matrix)
{
    vector<int> res;
    if (matrix.size() == 0)
        return res;

    int r1 = 0, r2 = matrix.size() - 1;
    int c1 = 0, c2 = matrix[0].size() - 1;

    while (r1 < r2 && c1 < c2)
    {
        for (int i = c1; i < c2; i++)
            res.push_back(matrix[r1][i]);
        for (int i = r1; i < r2; i++)
            res.push_back(matrix[i][c2]);
        for (int i = c2; i > c1; i--)
            res.push_back(matrix[r2][i]);
        for (int i = r2; i > r1; i--)
            res.push_back(matrix[i][c1]);

        c1++;
        c2--;
        r1++;
        r2--;
    }

    return res;
}

/********************/
/* 412. Fizz Buzz */
vector<string> fizzBuzz(int n)
{
    vector<string> res;
    bool mul3 = false, mul5 = false;

    for (int i = 1; i <= n; i++)
    {

        mul3 = false;
        mul5 = false;
        if (i % 3 == 0)
            mul3 = true;
        if (i % 5 == 0)
            mul5 = true;

        if (mul3 && mul5)
            res.push_back("FizzBuzz");
        else if (mul3)
            res.push_back("Fizz");
        else if (mul5)
            res.push_back("Buzz");
        else
            res.push_back(std::to_string(i));
    }

    return res;
}

/********************/
/* 46. Permutations */

void permute_recurse(vector<int> &nums, int begin, vector<vector<int>> &result)
{
    if (begin >= nums.size())
    {
        result.push_back(nums);
        return;
    }

    for (int i = begin; i < nums.size(); i++)
    {
        std::swap(nums[begin], nums[i]);
        permute_recurse(nums, begin + 1, result);
        //reset
        std::swap(nums[begin], nums[i]);
    }
}

vector<vector<int>> permute(vector<int> &nums)
{
    vector<vector<int>> result;
    permute_recurse(nums, 0, result);
    return result;
}

/********************/
/* 88. Merge Sorted Array */
void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
{
    vector<int> temp;
    for (int i = 0; i < m; i++)
        temp.push_back(nums1[i]);
    int i = 0, j = 0, k = 0;
    while (i < temp.size() && j < nums2.size())
    {
        if (temp[i] <= nums2[j])
        {
            nums1[k] = temp[i];
            i++;
        }
        else if (temp[i] > nums2[j])
        {
            nums1[k] = nums2[j];
            j++;
        }

        k++;
    }

    if (i < temp.size())
    {
        nums1[k] = temp[i];
        i++;
        k++;
    }
    if (j < nums2.size())
    {
        nums1[k] = nums2[i];
        j++;
        k++;
    }
}

/********************/
/* 160. Intersection of Two Linked Lists*/
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB)
{
    ListNode *pa = headA;
    ListNode *pb = headB;

    if (pa == NULL || pb == NULL)
        return NULL;

    while (pa != NULL && pb != NULL && pa != pb)
    {
        pa = pa->next;
        pb = pb->next;

        if (pa == pb)
            return pa;

        if (pa == NULL)
            pa = headB;
        if (pb == NULL)
            pb = headA;
    }

    return pa;
}
/********************/
/* 234. Palindrome Linked List*/
ListNode *temp;
bool isPalindrome(ListNode *head)
{
    temp = head;
    return check(head);
}

bool check(ListNode *node)
{
    if (node == NULL)
        return true;
    bool isPal = check(node->next) && (temp->val == node->val);
    temp = temp->next;
    return isPal;
}

/********************/
/* 23. Merge k Sorted Lists*/
ListNode *merge2Lists(ListNode *l1, ListNode *l2)
{
    if (l1 == NULL)
        return l2;
    if (l2 == NULL)
        return l1;

    if (l1->val <= l2->val)
    {
        l1->next = merge2Lists(l1->next, l2);
        return l1;
    }
    else
    {
        l2->next = merge2Lists(l1, l2->next);
        return l2;
    }
}

ListNode *mergeKLists(vector<ListNode *> &lists)
{
    if (lists.empty())
        return NULL;
    //straight solution
    while (lists.size() > 1)
    {
        lists.push_back(merge2Lists(lists[0], lists[1]));
        lists.erase(lists.begin());
        lists.erase(lists.begin());
    }

    return lists.front();
}

struct compare
{
    bool operator()(ListNode *n1, ListNode *n2)
    {
        return n1->val > n2->val;
    }
};

ListNode *mergeKListsPQ(vector<ListNode *> &lists)
{
    priority_queue<ListNode *, vector<ListNode *>, compare> q;
    for (auto l : lists)
    {
        if (l)
            q.push(l);
    }

    if (q.empty())
        return NULL;

    ListNode *result = q.top();
    ListNode *tail = result;
    q.pop();
    if (result->next)
        q.push(result->next);
    while (!q.empty())
    {
        tail->next = q.top();
        q.pop();
        tail = tail->next;
        if (tail->next)
            q.push(tail->next);
    }

    return result;
}

/********************/
/* 344. Reverse String*/
void reverseString(vector<char> &s)
{
    int i = 0, j = s.size() - 1;
    while (i < j)
    {
        s[i] = s[i] + s[j];
        s[j] = s[i] - s[j];
        s[j] = s[i] - s[j];

        i++;
        j--;
    }
}

/********************/
/* 79. Word Search*/
bool dfs_find(vector<vector<char>> &board, const string &word, int index, int i, int j)
{
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || index >= word.size() || board[i][j] == '\0')
        return false;

    if (word[index] != board[i][j])
        return false;
    if (index == word.size() - 1)
        return true;

    char temp = board[i][j];
    board[i][j] = '\0';

    if (dfs_find(board, word, index + 1, i + 1, j) || dfs_find(board, word, index + 1, i - 1, j) || dfs_find(board, word, index + 1, i, j + 1) || dfs_find(board, word, index + 1, i, j - 1))
        return true;

    board[i][j] = temp;
    return false;
}

bool exist(vector<vector<char>> &board, string word)
{
    if (word.empty())
        return true;
    if (board.size() == 0)
        return false;

    int r = board.size();
    int c = board[0].size();

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            bool res = dfs_find(board, word, 0, i, j);
            if (res)
                return res;
        }
    }

    return false;
}

/********************/
/* 203. Remove Linked List Elements*/
ListNode *removeElements(ListNode *head, int val)
{
    ListNode **ppl = &head;

    while (*ppl != NULL)
    {
        if ((*ppl)->val == val)
        {
            *ppl = (*ppl)->next;
        }
        else
        {
            ppl = &(*ppl)->next;
        }
    }

    return head;
}
/********************/
/* 103. Binary Tree Zigzag Level Order Traversal*/
vector<vector<int>> zigzagLevelOrder(TreeNode *root)
{
    vector<vector<int>> result;
    if (NULL == root)
        return result;

    stack<TreeNode *> st1;
    stack<TreeNode *> st2;
    int lvl = 0;

    st1.push(root);
    while (1)
    {
        if (st1.empty() && st2.empty())
            break;

        vector<int> tempVec;

        while (!st1.empty())
        {
            TreeNode *pNode = st1.top();
            tempVec.push_back(pNode->val);
            if (pNode->left)
                st2.push(pNode->left);
            if (pNode->right)
                st2.push(pNode->right);
            st1.pop();
        }

        if (tempVec.size() > 0)
        {
            result.push_back(tempVec);
            tempVec.clear();
        }

        while (!st2.empty())
        {
            TreeNode *pNode = st2.top();
            tempVec.push_back(pNode->val);
            if (pNode->left)
                st1.push(pNode->left);
            if (pNode->right)
                st1.push(pNode->right);
            st2.pop();
        }

        if (tempVec.size() > 0)
            result.push_back(tempVec);
    }

    return result;
}
/********************/
/* 215. Kth Largest Element in an Array*/
int findKthLargest(vector<int> &nums, int k)
{
    if (0 == nums.size())
        return 0;

    priority_queue<int, vector<int>, std::greater<int>> pq;
    for (int i = 0; i < nums.size(); i++)
    {
        pq.push(nums[i]);
        if (pq.size() > k)
            pq.pop();
    }

    return pq.top();
}
/********************/
/* 138. Copy List with Random Pointer*/
class Node
{
  public:
    int val;
    Node *next;
    Node *random;

    Node() {}

    Node(int _val, Node *_next, Node *_random)
    {
        val = _val;
        next = _next;
        random = _random;
    }
};

map<Node *, Node *> clone_node_map;
Node *get_clone_node(Node *oldNode)
{
    if (NULL == oldNode)
        return NULL;
    if (clone_node_map.find(oldNode) != clone_node_map.end())
    {
        return clone_node_map[oldNode];
    }
    else
    {
        clone_node_map[oldNode] = new Node(oldNode->val, NULL, NULL);
        return clone_node_map[oldNode];
    }
}

Node *copyRandomList(Node *head)
{
    if (NULL == head)
        return head;

    Node *pNewHead = new Node(head->val, NULL, NULL);
    clone_node_map[head] = pNewHead;
    Node *pOldHead = head;

    while (pOldHead)
    {
        pNewHead->next = get_clone_node(pOldHead->next);
        pNewHead->random = get_clone_node(pOldHead->random);

        pOldHead = pOldHead->next;
        pNewHead = pNewHead->next;
    }

    return clone_node_map[head];
}
/********************/
/* 98. Validate Binary Search Tree*/
void get_inorder(TreeNode *root, vector<TreeNode *> &vecInorder)
{
    if (root == NULL)
        return;

    get_inorder(root->left, vecInorder);
    vecInorder.push_back(root);
    get_inorder(root->right, vecInorder);
}

bool isValidBST(TreeNode *root)
{
    if (NULL == root)
        return true;

    vector<TreeNode *> vecInorder;
    get_inorder(root, vecInorder);

    for (int i = 0; i < vecInorder.size() - 1; i++)
    {
        if (vecInorder[i]->val >= vecInorder[i + 1]->val)
            return false;
    }

    return true;
}
/********************/
/* 226. Invert Binary Tree*/
TreeNode *invertTree(TreeNode *root)
{
    if (root == NULL)
        return NULL;
    TreeNode *pTemp = root->left;
    root->left = invertTree(root->right);
    root->right = invertTree(pTemp);

    return root;
}

/********************/
/* 543. Diameter of Binary Tree*/
int calTreeNodeLen(TreeNode *root, int &max)
{
    if (NULL == root)
        return 0;

    int left_len = calTreeNodeLen(root->left, max);
    int right_len = calTreeNodeLen(root->right, max);

    max = std::max(max, left_len + right_len + 1);
    return std::max(left_len, right_len) + 1;
}

int diameterOfBinaryTree(TreeNode *root)
{
    if (NULL == root)
        return 0;

    int max = 1;
    calTreeNodeLen(root, max);

    return max - 1;
}
/********************/
/* 199. Binary Tree Right Side View*/
vector<int> rightSideView(TreeNode *root)
{
    vector<int> res;
    if (NULL == root)
        return res;

    queue<TreeNode *> node_queue;
    node_queue.push(root);

    while (!node_queue.empty())
    {
        int len = node_queue.size();
        for (int i = 0; i < len; i++)
        {
            TreeNode *pTemp = node_queue.front();
            if (pTemp->left)
                node_queue.push(pTemp->left);
            if (pTemp->right)
                node_queue.push(pTemp->right);
            if (i == len - 1)
                res.push_back(pTemp->val);

            node_queue.pop();
        }
    }

    return res;
}
/********************/
/* 94. Binary Tree Inorder Traversal*/
vector<int> inorderTraversal(TreeNode *root)
{
    vector<int> res;
    stack<TreeNode *> st;
    TreeNode *pTempRoot = root;

    while (pTempRoot || !st.empty())
    {
        while (pTempRoot)
        {
            st.push(pTempRoot);
            pTempRoot = pTempRoot->left;
        }

        pTempRoot = st.top();
        res.push_back(pTempRoot->val);
        st.pop();

        pTempRoot = pTempRoot->right;
    }

    return res;
}

/********************/
/* 124. Binary Tree Maximum Path Sum*/
int getMaxSum(TreeNode *root, int &max)
{
    if (root == NULL)
        return 0;

    int left_max = getMaxSum(root->left, max);
    left_max = left_max > 0 ? left_max : 0;
    int right_max = getMaxSum(root->right, max);
    right_max = right_max > 0 ? right_max : 0;

    int temp = root->val + left_max + right_max;
    if (temp > max)
        max = temp;

    return root->val + std::max(left_max, right_max);
}

int maxPathSum(TreeNode *root)
{
    int max = INT_MIN;
    getMaxSum(root, max);
    return max;
}
/********************/
/* 236. Lowest Common Ancestor of a Binary Tree*/
bool find_path(TreeNode *root, TreeNode *pTargetNode, vector<TreeNode *> &vecPath)
{
    if (NULL == root)
        return false;

    vecPath.push_back(root);
    if (root == pTargetNode)
        return true;
    else
    {
        bool bFindLeft = find_path(root->left, pTargetNode, vecPath);
        if (!bFindLeft)
        {
            bool bFindRight = find_path(root->right, pTargetNode, vecPath);
            if (!bFindRight)
            {
                vecPath.pop_back();
                return false;
            }
        }
    }

    return true;
}

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
{
    if (NULL == root)
        return NULL;
    vector<TreeNode *> vec_p, vec_q;
    bool bLeft = find_path(root, p, vec_p);
    bool bRight = find_path(root, q, vec_q);
    if (bLeft && bRight)
    {
        int i = 0, j = 0;
        TreeNode *lca = root;
        while (i < vec_p.size() && j < vec_q.size())
        {
            if (vec_p[i] == vec_q[j])
            {
                lca = vec_p[i];
                i++;
                j++;
            }
            else
                break;
        }

        return lca;
    }

    return NULL;
}
/********************/
/* 295. Find Median from Data Stream*/
class MedianFinder
{
  public:
    /** initialize your data structure here. */
    MedianFinder()
    {
    }

    void addNum(int num)
    {
        data.insert(std::lower_bound(data.begin(), data.end(), num), num);
    }

    double findMedian()
    {
        int n = data.size();
        if (n & 1)
            return data[n / 2];
        else
        {
            return ((double)data[n / 2] + (double)data[n / 2 - 1]) / 2.0;
        }
    }

  private:
    vector<int> data;
};
class MedianFinder2Heaps //with two heap
{
  public:
    /** initialize your data structure here. */
    MedianFinder2Heaps()
    {
    }

    void addNum(int num)
    {
        low.push(num);

        high.push(low.top());
        low.pop();

        if (low.size() < high.size())
        {
            low.push(high.top());
            high.pop();
        }
    }

    double findMedian()
    {
        return low.size() > high.size() ? low.top() : (low.top() + high.top()) * 0.5;
    }

  private:
    priority_queue<int> low;
    priority_queue<int, vector<int>, std::greater<int>> high;
};
/********************/
/* 301. Remove Invalid Parentheses */
vector<string> removeInvalidParentheses(string s)
{
    map<char, int> par_map;
    par_map['('] = -1;
    par_map[')'] = 1;

    int sum = 0;
    for (char c : s)
    {
        if (par_map.find(c) != par_map.end())
        {
            sum += par_map[c];
            if (c == ')')
            {
                //if(sum > 0)//delete ')'
            }
        }
    }
}

/********************/
/* 17. Letter Combinations of a Phone Number*/
map<char, string> num_map;
void constructStr(const string &digits, int index, string strRes, vector<string> &vecRes)
{
    if (strRes.size() == digits.size())
    {
        vecRes.push_back(strRes);
        return;
    }

    char cur_ch = digits[index];
    const string &cur_str = num_map[cur_ch];

    for (char c : cur_str)
    {
        constructStr(digits, index + 1, strRes + c, vecRes);
    }
}

vector<string> letterCombinations(string digits)
{
    vector<string> res;
    if (digits.empty())
        return res;

    num_map['2'] = "abc";
    num_map['3'] = "def";
    num_map['4'] = "ghi";
    num_map['5'] = "jkl";
    num_map['6'] = "mno";
    num_map['7'] = "pqrs";
    num_map['8'] = "tuv";
    num_map['9'] = "wxyz";

    constructStr(digits, 0, "", res);

    return res;
}
/********************/
/* 380. Insert Delete GetRandom O(1)*/
class RandomizedSet
{
  public:
    /** Initialize your data structure here. */
    RandomizedSet()
    {
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val)
    {
        if (idx_map.find(val) != idx_map.end())
            return false;
        nums.emplace_back(val);
        idx_map[val] = nums.size() - 1;
        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val)
    {
        if (idx_map.find(val) == idx_map.end())
            return false;
        int last = nums.back();
        idx_map[last] = idx_map[val];
        nums[idx_map[last]] = last;
        nums.pop_back();
        idx_map.erase(val);
        return true;
    }

    /** Get a random element from the set. */
    int getRandom()
    {
        return nums[rand() % nums.size()];
    }

  private:
    vector<int> nums;
    unordered_map<int, int> idx_map;
};

/********************/
/* 139. Word Break*/

/********************/
/* 946. Validate Stack Sequences*/
bool validateStackSequences(vector<int>& pushed, vector<int>& popped)
{
    if(pushed.size() == 0 && popped.size() == 0)
        return true;

    if(pushed.size() == 0 || popped.size() == 0)
        return false;

    bool bValid = true;
    
    stack<int> st_data;
    int j = 0;
    for(int i = 0; i < popped.size(); i++)
    {
        while(st_data.size() == 0 || st_data.top() != popped[i]) 
        {
            if(j == pushed.size())
                break;

            st_data.push(pushed[j]);            
            j++;
        }

        if(st_data.top() != popped[i])
        {
            bValid = false;
            break;
        }
        
        st_data.pop();
    }

    return bValid; 
}

/********************/
/* 255. Verify Preorder Sequence in Binary Search Tree*/
bool verifyPreorderHelper(vector<int>& preorder, int s, int e)
{
    if(preorder.size() == 0)
        return true;
    if(s >= e)
        return true;

    int nRoot = preorder[s]; //root value
    int left = s + 1; 
    for(;left < e; left++)
    {
        if(preorder[left] > nRoot)
            break;
    }

    for(int j = left; j < e; j++)
    {
        if(preorder[j] < nRoot)
            return false;
    }

    bool bLeftValid = verifyPreorderHelper(preorder, s + 1, left);
    if(bLeftValid)
    {
        bool bRightValid = verifyPreorderHelper(preorder, left, e);
        return bRightValid;
    }

    return false;
}

bool verifyPreorder(vector<int>& preorder)
{
    return verifyPreorderHelper(preorder, 0, preorder.size());
}

/********************/
/* 113. Path Sum II*/
void pathSumHelper(TreeNode *root, int sum, int curSum, vector<int>& vecTemp, vector<vector<int> >& res)
{
    if(NULL == root)
        return;

    vecTemp.push_back(root->val);
    curSum += root->val;
    bool bLeaf = !root->left && !root->right;
    if(bLeaf && curSum == sum)
    {
        res.push_back(vecTemp); 
    }
    else
    {
        pathSumHelper(root->left, sum ,curSum, vecTemp, res);
        pathSumHelper(root->right, sum ,curSum, vecTemp, res);
    }

    vecTemp.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int sum) 
{
    vector<vector<int> > res;
    if(NULL == root)
        return res;
    vector<int> temp;
    int curSum = 0;
    pathSumHelper(root, sum, curSum, temp, res);
    return res;
}

/********************/
/* 426. Convert Binary Search Tree to Sorted Doubly Linked List*/
void treeToDoublyListHelper(TreeNode *root, TreeNode **ppLastNode)
{
    if(NULL == root)
        return;

    if(root->left)
        treeToDoublyListHelper(root->left, ppLastNode);

    root->left = *ppLastNode;
    if(*ppLastNode)
        (*ppLastNode)->right = root;

    *ppLastNode = root;

    if(root->right)
        treeToDoublyListHelper(root->right, ppLastNode);
}

TreeNode* treeToDoublyList(TreeNode* root) 
{
    if(NULL == root)
        return NULL;
    TreeNode *ppLastNode = NULL;
    treeToDoublyListHelper(root, &ppLastNode);

    TreeNode *pCpLastNode = ppLastNode;
    while(1)
    {
        TreeNode *pLeft = (ppLastNode)->left;
        if(pLeft)
            (ppLastNode) = pLeft;
        else
        {
            ppLastNode->left = pCpLastNode;
            pCpLastNode->right = ppLastNode;
            return ppLastNode;
        }
    }

    return NULL;
}
/********************/
/* 136. Single Number*/
int singleNumber(vector<int>& nums) 
{
    if(nums.size() == 0)
        return 0;

    int res = nums[0];
    for(int i = 1; i < nums.size(); i++)
    {
        res ^= nums[i];
    }

    return res;
}
/********************/
/* 264. Ugly Number II*/
int nthUglyNumber(int n)
{
    if(0 >= n)
        return 0;
    if(1 == n)
        return 1;
    vector<int> vecUgly(n);

    int p2 = 0, p3 = 0, p5 = 0;
    vecUgly[0] = 1;

    for(int i = 1; i < n; i++)
    {
        vecUgly[i] = std::min(vecUgly[p2] * 2, std::min(vecUgly[p3] * 3, vecUgly[p5] * 5));
        if(vecUgly[i] == vecUgly[p2] * 2)
            p2++;
        if(vecUgly[i] == vecUgly[p3] * 3)
            p3++;
        if(vecUgly[i] == vecUgly[p5] * 5)
            p5++;
    }

    return vecUgly[n - 1];
}
/********************/
/* 179. Largest Number*/
string largestNumber(vector<int>& nums)
{
    string result = "0";
    if(0 == nums.size())
        return result;

    std::sort(nums.begin(), nums.end(), [&](int a, int b)
    {
        string tempAb = std::to_string(a) + std::to_string(b);
        string tempBa = std::to_string(b) + std::to_string(a);
        return tempAb > tempBa;
    });

    for(int i = 0; i < nums.size(); i++)
    {
        if(result == "0")
            result = "";
        result += std::to_string(nums[i]);
    }
    return result;
}
/********************/
/* 387. First Unique Character in a String*/
int firstUniqChar(string s)
{
    int ch_map[256];
    for(int i = 0; i < 256; i++)
        ch_map[i] = 0;
    for(char c : s)
    {
        ch_map[c]++;
    }

    for(char c : s)
        if(ch_map[c] == 1)
            return s.find_first_of(c, 0);

    return -1;
}

/********************/
/* 49. Group Anagrams*/
vector<vector<string>> groupAnagrams(vector<string>& strs) 
{
    unordered_map<string, vector<string>> res_map;
    for(string& str : strs)
    {
        string temp = str;
        std::sort(temp.begin(), temp.end());
        if(res_map.find(temp) != res_map.end())
            res_map[temp].push_back(str);
        else
        {
            vector<string> vecTemp;
            vecTemp.push_back(str);
            res_map[temp] = vecTemp;
        }
    }

    vector<vector<string>> result;
    for(auto it = res_map.begin(); it != res_map.end(); it++)
       result.push_back(it->second); 

    return result;
}

/********************/
/* 34. Find First and Last Position of Element in Sorted Array*/
int find_first_target_num(vector<int>& nums, int target, int start, int end)
{
    if(start > end)
        return -1;
    int mid = (start + end) / 2;
    if(nums[mid] == target)
    {
        if(mid == 0 || nums[mid - 1] != target)
            return mid;
        else
            end = mid - 1;
    }
    else if(nums[mid] > target)
        end = mid - 1;
    else
        start = mid + 1;
    
    return find_first_target_num(nums, target, start, end);
}

int find_last_target_num(vector<int>& nums, int target, int start, int end)
{
    if(start > end)
        return -1;
    int mid = (start + end) / 2;
    if(nums[mid] == target)
    {
        if(mid == nums.size() - 1 || nums[mid + 1] != target)
            return mid;
        else
            start = mid + 1;
    }
    else if(nums[mid] > target)
        end = mid - 1;
    else
        start = mid + 1;
    
    return find_last_target_num(nums, target, start, end);
}

vector<int> searchRange(vector<int>& nums, int target)
{
    vector<int> res(2, -1);
    if(nums.size() == 0)
    {
        return res;
    }

    int s = find_first_target_num(nums, target, 0, nums.size() - 1);
    int e = find_last_target_num(nums, target, 0, nums.size() - 1);

    res[0] = s;    
    res[1] = e;    
    return res;
}
/********************/
/* 297. Serialize and Deserialize Binary Tree*/
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string result = "";
        if(root == NULL)
        {
            result = "null,";
            return result;
        }

        result += std::to_string(root->val);
        result += ",";

        result += serialize(root->left);
        result += serialize(root->right);

        std::cout << result << endl;
        return result;
    }

    void construct_tree(list<string>& vecNodes, TreeNode **root)
    {
        if(vecNodes.size() == 0)
            return;

        if(vecNodes.front() == "null") 
        {
            vecNodes.pop_front();
            *root = NULL;
            return;
        }

        int val = std::atoi(vecNodes.front().c_str());
        *root = new TreeNode(val);
        vecNodes.pop_front();

        construct_tree(vecNodes, &((*root)->left));
        construct_tree(vecNodes, &((*root)->right));
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.empty())
            return NULL;

        list<string> vecNodes;
        std::stringstream ss(data);        
        string token;
        while (getline(ss,token, ','))
        {
            vecNodes.push_back(token); 
        }

        if(vecNodes.size() == 0)
            return NULL;
        
        TreeNode *root = NULL;
        construct_tree(vecNodes, &root);

        return root;
    }
};
/********************/
/* 260. Single Number III*/
int find_1bit_index(int i)
{
    int temp = 1;
    int idx = 0;
    while(temp & i == 0)
    {
        temp = temp << 1;
        idx++;
    }

    return idx;
}

bool is_idx_bit_1(int i, int idx)
{
    int temp = 1 << idx;
    return temp & i; 
}

vector<int> singleNumberIII(vector<int>& nums) 
{
    vector<int> result;    
    if(nums.size() == 0)
        return result;
    int temp = 0; 
    for(int i : nums)
        temp ^= i;

    int targetIdx = find_1bit_index(temp);
    int m = 0, n = 0;
    for(int i : nums)     
    {
        if(is_idx_bit_1(i, targetIdx))
            m ^= i;
        else
            n ^= i;
    }

    result.push_back(m);
    result.push_back(n);
    return result;
}
/********************/
/* 137. Single Number II*/
int singleNumberII(vector<int>& nums) 
{
    const int BIT_COUNT = 32;
    int bits[BIT_COUNT];
    for(int i : nums)
    {
        int64_t bitInit = 1;
        for(int j = 0; j < BIT_COUNT; j++)
        {
            bits[j] += (bitInit & i);
            bitInit = bitInit << 1;
        }
    }

    int result = 0;
    for(int i = BIT_COUNT - 1; i >= 0; i--)
    {
        result = result << 1;
        result += bits[i] % 3;
    }

    return result;
}
/********************/
/* 239. Sliding Window Maximum*/
vector<int> maxSlidingWindow(vector<int>& nums, int k)
{
    vector<int> result;
    if(nums.size() == 0)
        return result;
    deque<int> index;    

    for(int i = 0; i < k; i++)
    {
        while(!index.empty() && nums[i] > nums[index.back()])
            index.pop_back();
        index.push_back(i);
    }

    for(int i = k; i < nums.size(); i++)
    {
        result.push_back(nums[index.front()]);

        while(!index.empty() && nums[i] > nums[index.back()])
            index.pop_back();
        if(!index.empty() && index.front() < (i - k))
            index.pop_front();

        index.push_back(i);
    }

    result.push_back(nums[index.front()]);
    return result;
}

/********************/

int main(int argc, char **argv)
{
    /*vector<int> arr = {2, 7, 11, 15};
int target = 9;
vector<int> result;
result = twoSum(arr, target);
for(int i : result)
{
    cout << " " << i << endl;
}*/
    int a = -321;
    cout << "==>" << reverse(a);

    return 0;
}