var Result = angular.module('Result', []);

Result.controller('webcontrol', ['$http', '$scope', function ($http, $scope) { 
    var refresh = function () {
        $http.get("/images").success(function (response) {
            $scope.imageList = response;
        });
    }
    refresh(); 
}]);